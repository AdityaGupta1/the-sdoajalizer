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

void NodeEvaluator::setOutputTexture(Texture* texture)
{
    this->outputTexture = texture;
}

bool NodeEvaluator::hasOutputTexture()
{
    return this->outputTexture != nullptr;
}

// this might be unnecessarily complicated but strikes a good balance between memory usage and performance
// - caching everything would lead to significantly higher memory usage
// - not caching means bad performance on editing nodes later on in a chain
void NodeEvaluator::setChangedNode(Node* changedNode)
{
    // do this no matter what for changedNode so user can use its output without reevaluating
    for (auto& outputPin : changedNode->outputPins)
    {
        outputPin.deleteCache();
        outputPin.prepareForCache();
    }

    if (changedNode == this->changedNode)
    {
        return;
    }

    this->changedNode = changedNode;

    // reachability search from changed node
    // delete cache of any pins reachable from this node
    std::queue<Node*> frontier;
    std::unordered_set<Node*> visited;
    frontier.push(this->changedNode);
    visited.insert(this->changedNode);
    while (!frontier.empty())
    {
        Node* node = frontier.front();
        frontier.pop();

        for (auto& outputPin : node->outputPins)
        {
            for (const auto& edge : outputPin.getEdges())
            {
                Pin* otherInputPin = edge->endPin;
                if (node != this->changedNode)
                {
                    outputPin.deleteCache();
                }

                Node* otherNode = otherInputPin->getNode();
                if (!visited.contains(otherNode))
                {
                    visited.insert(otherNode);
                    frontier.push(otherNode);
                }
            }
        }

        this->nodesToDeleteCache.erase(node); // no need to delay cache deletion since caches were just deleted
    }

    if (!visited.contains(this->outputNode))
    {
        return;
    }

    // do one step of reachability search backwards to find where to prepare caches
    std::queue<Node*> frontierBackwards;
    std::unordered_set<Node*> visitedBackwards = visited;
    for (const auto& node : visited)
    {
        for (const auto& inputPin : node->inputPins)
        {
            for (const auto& edge : inputPin.getEdges()) // should be at most 1 edge
            {
                Node* otherNode = edge->startPin->getNode();
                if (!visitedBackwards.contains(otherNode))
                {
                    visitedBackwards.insert(otherNode);
                    frontierBackwards.push(otherNode);

                    for (auto& otherOutputPin : otherNode->outputPins)
                    {
                        otherOutputPin.prepareForCache();
                    }
                }
            }
        }
    }

    // finish reachability search and prepare caches for deletion
    while (!frontierBackwards.empty())
    {
        Node* node = frontierBackwards.front();
        frontierBackwards.pop();

        for (const auto& inputPin : node->inputPins)
        {
            for (const auto& edge : inputPin.getEdges()) // should be at most 1 edge
            {
                Node* otherNode = edge->startPin->getNode();
                if (!visitedBackwards.contains(otherNode))
                {
                    visitedBackwards.insert(otherNode);
                    frontierBackwards.push(otherNode);

                    nodesToDeleteCache.insert(otherNode);
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
        Node* node = frontier.front();
        frontier.pop();

        int indegree = 0;
        for (const auto& inputPin : node->inputPins)
        {
            for (const auto& edge : inputPin.getEdges()) // should be at most 1 edge
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

        indegrees[node] = indegree;
        if (indegree == 0)
        {
            nodesWithIndegreeZero.push(node);
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
        node->evaluate(); // will cache textures in pins if necessary when calling propagateTexture()
        node->clearInputTextures();

        for (auto& tex : requestedTextures)
        {
            --tex->numReferences;
        }
        requestedTextures.clear();
    }

    for (const auto& node : nodesToDeleteCache)
    {
        for (auto& outputPin : node->outputPins)
        {
            outputPin.deleteCache();
        }
    }

    nodesToDeleteCache.clear();

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
}
