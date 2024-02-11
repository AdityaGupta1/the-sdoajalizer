#include "node_evaluator.hpp"

#include <stack>
#include <queue>
#include <unordered_map>

#include "common.hpp"

#include <glm/gtx/string_cast.hpp>
#include <iostream>

NodeEvaluator::NodeEvaluator(glm::ivec2 outputResolution)
    : outputResolution(outputResolution), viewerTex1(-1), viewerTex2(-1)
{}

NodeEvaluator::~NodeEvaluator()
{
    for (const auto& [resolution, resTextures] : this->textures)
    {
        for (const auto& texture : resTextures)
        {
            CUDA_CHECK(cudaFree(texture->dev_pixels));
        }
    }
}

void NodeEvaluator::init()
{
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &viewerTex1);
    glBindTexture(GL_TEXTURE_2D, viewerTex1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glActiveTexture(GL_TEXTURE1);
    glGenTextures(1, &viewerTex2);
    glBindTexture(GL_TEXTURE_2D, viewerTex2);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
}

void NodeEvaluator::setOutputNode(Node* outputNode)
{
    this->outputNode = outputNode;
}

void NodeEvaluator::setSelectedNode(Node* selectedNode)
{
    if (this->selectedNode == selectedNode)
    {
        return;
    }

    if (this->selectedNode != nullptr)
    {
        this->selectedNode->setIsSelected(false);
    }

    this->selectedNode = selectedNode;

    if (this->selectedNode == nullptr)
    {
        selectedTexture = nullptr;
        return;
    }

    this->selectedNode->setIsSelected(true);

    selectedTexture = nullptr;
    evaluateFrom(this->selectedNode);

    if (selectedTexture != nullptr)
    {
        setGlTexture(selectedTexture, GL_TEXTURE0);
    }

    cleanupTextureReferences();
}

Texture* NodeEvaluator::requestTexture()
{
    return this->requestTexture(this->outputResolution);
}

Texture* NodeEvaluator::requestTexture(glm::ivec2 resolution)
{
    if (this->textures.contains(resolution))
    {
        for (const auto& texture : this->textures[resolution])
        {
            if (texture->numReferences == 0)
            {
                return texture.get();
            }
        }
    }
    else
    {
        this->textures[resolution] = std::vector<std::unique_ptr<Texture>>();
    }

    auto tex = std::make_unique<Texture>();
    CUDA_CHECK(cudaMalloc(&tex->dev_pixels, resolution.x * resolution.y * sizeof(glm::vec4)));
    tex->resolution = resolution;

    Texture* texPtr = tex.get();
    this->textures[resolution].push_back(std::move(tex));
    return texPtr;
}

void NodeEvaluator::setOutputTexture(Texture* texture)
{
    this->outputTexture = texture;
}

void NodeEvaluator::setSelectedTexture(Texture* texture)
{
    this->selectedTexture = texture;
}

Texture* NodeEvaluator::getSelectedTexture()
{
    return this->selectedTexture;
}

void NodeEvaluator::evaluateFrom(Node* node)
{
    std::stack<Node*> nodesWithIndegreeZero; // using a stack to allow for a more depth-first topological sort?
                                             // might mean better memory usage during evaluation, idk
    std::unordered_map<Node*, int> indegrees;

    std::queue<Node*> frontier;
    std::unordered_set<Node*> visited;
    frontier.push(node);
    visited.insert(node);
    while (!frontier.empty())
    {
        Node* node = frontier.front();
        frontier.pop();

        int indegree = 0;
        for (const auto& inputPin : node->inputPins)
        {
            for (const auto& edge : inputPin.getEdges())
            {
                ++indegree;
                Node* otherNode = edge->startPin->getNode();
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
        node->evaluate();
        node->clearInputTextures();
    }

    cudaDeviceSynchronize();
}

void NodeEvaluator::evaluate()
{
    selectedTexture = nullptr;

    evaluateFrom(outputNode);

    if (selectedTexture == nullptr && selectedNode != nullptr)
    {
        evaluateFrom(this->selectedNode);
    }

    if (outputTexture == selectedTexture && outputTexture != nullptr)
    {
        setGlTexture(outputTexture, GL_TEXTURE1, GL_TEXTURE0);
    }
    else
    {
        if (outputTexture != nullptr)
        {
            setGlTexture(outputTexture, GL_TEXTURE1);
        }

        if (selectedTexture != nullptr)
        {
            setGlTexture(selectedTexture, GL_TEXTURE0);
        }
    }

    cleanupTextureReferences();
}

void NodeEvaluator::setGlTexture(Texture* texture, GLenum glTexture1, GLenum glTexture2)
{
    float* host_pixels;
    int sizeBytes = texture->resolution.x * texture->resolution.y * sizeof(glm::vec4);
    CUDA_CHECK(cudaMallocHost(&host_pixels, sizeBytes));

    CUDA_CHECK(cudaMemcpy(
        host_pixels,
        texture->dev_pixels,
        sizeBytes,
        cudaMemcpyDeviceToHost
    ));

    // TODO: CUDA/OpenGL interop (at least for the output node; that node may need to request a special texture from NodeEvaluator)
    // TODO: replace with glTexSubImage2D?
    glActiveTexture(glTexture1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, texture->resolution.x, texture->resolution.y, false, GL_RGBA, GL_FLOAT, host_pixels);

    if (glTexture2 != 0)
    {
        glActiveTexture(glTexture2);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, texture->resolution.x, texture->resolution.y, false, GL_RGBA, GL_FLOAT, host_pixels);
    }

    CUDA_CHECK(cudaFreeHost(host_pixels));
}

void NodeEvaluator::cleanupTextureReferences()
{
    for (const auto& [res, resTextures] : this->textures)
    {
        for (const auto& tex : resTextures)
        {
            tex->numReferences = 0;
        }
    }
}
