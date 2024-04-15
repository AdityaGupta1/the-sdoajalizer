#include "node_evaluator.hpp"

#include <stack>
#include <queue>
#include <unordered_map>

NodeEvaluator::NodeEvaluator(glm::ivec2 outputResolution)
    : outputResolution(outputResolution), viewerTex(-1)
{}

NodeEvaluator::~NodeEvaluator()
{
    for (const auto& [res, resTextures] : this->textures)
    {
        for (const auto& texture : resTextures)
        {
            if (texture->dev_pixels != nullptr)
            {
                CUDA_CHECK(cudaFree(texture->dev_pixels));
            }
        }
    }
}

void NodeEvaluator::init()
{
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &viewerTex);
    glBindTexture(GL_TEXTURE_2D, viewerTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
}

void NodeEvaluator::setOutputNode(Node* outputNode)
{
    this->outputNode = outputNode;
}

Texture* NodeEvaluator::requestTexture(glm::ivec2 resolution)
{
    if (this->textures.contains(resolution))
    {
        for (const auto& texture : this->textures[resolution])
        {
            if (texture->numReferences == 0)
            {
                ++texture->numReferences;
                requestedTextures.push_back(texture.get());
                return texture.get();
            }
        }
    }
    else
    {
        this->textures[resolution] = std::vector<std::unique_ptr<Texture>>();
    }

    auto tex = std::make_unique<Texture>();
    
    if (resolution.x != 0)
    {
        CUDA_CHECK(cudaMalloc(&tex->dev_pixels, resolution.x * resolution.y * sizeof(glm::vec4)));
        tex->resolution = resolution;
    }

    Texture* texPtr = tex.get();
    this->textures[resolution].push_back(std::move(tex));

    ++texPtr->numReferences;
    requestedTextures.push_back(texPtr);

    return texPtr;
}

Texture* NodeEvaluator::requestTexture()
{
    return this->requestTexture(this->outputResolution);
}

Texture* NodeEvaluator::requestSingleColorTexture()
{
    return this->requestTexture(glm::ivec2(0));
}

Texture* NodeEvaluator::getOutputTexture() const
{
    return this->outputTexture;
}

void NodeEvaluator::setOutputTexture(Texture* texture)
{
    this->outputTexture = texture;
}

bool NodeEvaluator::hasOutputTexture() const
{
    return this->outputTexture != nullptr;
}

// this might be unnecessarily complicated but strikes a good balance between memory usage and performance
// - caching everything would lead to significantly higher memory usage
// - not caching means bad performance on editing nodes later on in a chain
void NodeEvaluator::setChangedNode(Node* changedNode)
{
    // delete cache of any pins reachable from changed node
    std::queue<Node*> frontier;
    std::unordered_set<Node*> visited;
    frontier.push(changedNode);
    visited.insert(changedNode);
    while (!frontier.empty())
    {
        Node* thisNode = frontier.front();
        frontier.pop();

        for (auto& thisOutputPin : thisNode->outputPins)
        {
            thisOutputPin.deleteCache();

            for (const auto& edge : thisOutputPin.getEdges())
            {
                Node* otherNode = edge->endPin->getNode();
                if (!visited.contains(otherNode))
                {
                    visited.insert(otherNode);
                    frontier.push(otherNode);
                }
            }
        }
    }
}

void NodeEvaluator::evaluate()
{
    std::stack<Node*> nodesWithIndegreeZero; // using a stack to allow for a more depth-first topological sort?
                                             // might mean better memory usage during evaluation, idk
    std::unordered_map<Node*, int> indegrees;

    std::queue<Node*> frontier;
    std::unordered_set<Node*> visited;
    frontier.push(this->outputNode);
    visited.insert(this->outputNode);
    while (!frontier.empty())
    {
        Node* thisNode = frontier.front();
        frontier.pop();

        int indegree = 0;
        for (const auto& thisInputPin : thisNode->inputPins)
        {
            for (const auto& edge : thisInputPin.getEdges()) // should be at most 1 edge
            {
                Pin* otherOutputPin = edge->startPin;

                if (otherOutputPin->getCacheState() == PinCacheState::CACHED)
                {
                    continue;
                }

                ++indegree;

                Node* otherNode = otherOutputPin->getNode();
                if (!visited.contains(otherNode))
                {
                    visited.insert(otherNode);
                    frontier.push(otherNode);
                }
            }
        }

        indegrees[thisNode] = indegree;
        if (indegree == 0)
        {
            nodesWithIndegreeZero.push(thisNode);
        }
    }

    // TODO: check for cycles in the above search (probably need to convert to DFS)

    std::vector<Node*> topoSortedNodes;
    while (!nodesWithIndegreeZero.empty())
    {
        Node* node = nodesWithIndegreeZero.top();
        nodesWithIndegreeZero.pop();

        topoSortedNodes.push_back(node);

        for (const auto& outputPin : node->outputPins)
        {
            for (const auto& edge : outputPin.getEdges())
            {
                Node* otherNode = edge->endPin->getNode();
                if (--indegrees[otherNode] == 0)
                {
                    nodesWithIndegreeZero.push(otherNode);
                }
            }
        }
    }

    for (const auto& node : topoSortedNodes)
    {
        if (node->isExpensive)
        {
            for (auto& outputPin : node->outputPins)
            {
                outputPin.prepareForCache(); // does nothing if cache already exists
            }
        }

        node->evaluate(); // will cache textures in pins if necessary when calling propagateTexture()
        node->clearInputTextures();

        for (auto& tex : requestedTextures)
        {
            --tex->numReferences;
        }
        requestedTextures.clear();
    }

    cudaDeviceSynchronize();

    if (outputTexture == nullptr)
    {
        return;
    }

    float* host_pixels;
    int sizeBytes = outputResolution.x * outputResolution.y * sizeof(glm::vec4);
    CUDA_CHECK(cudaMallocHost(&host_pixels, sizeBytes));
    CUDA_CHECK(cudaMemcpy(host_pixels, outputTexture->dev_pixels, sizeBytes, cudaMemcpyDeviceToHost));

    // TODO: CUDA/OpenGL interop (may need to request a special texture from NodeEvaluator)
    // TODO: replace with glTexSubImage2D?
    glActiveTexture(GL_TEXTURE0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, outputResolution.x, outputResolution.y, false, GL_RGBA, GL_FLOAT, host_pixels);

    CUDA_CHECK(cudaFreeHost(host_pixels));

    int numTextures = 0;
    for (const auto& [res, textures] : this->textures)
    {
        numTextures += textures.size();
    }
    printf("number of allocated textures: %d\n", numTextures);
}
