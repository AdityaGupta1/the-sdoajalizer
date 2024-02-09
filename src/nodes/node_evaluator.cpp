#include "node_evaluator.hpp"

#include <stack>
#include <queue>
#include <unordered_map>

#include "../common.hpp"

#include <iostream>

NodeEvaluator::NodeEvaluator()
{}

NodeEvaluator::~NodeEvaluator()
{
    for (const auto& [resolution, resTextures] : this->textures)
    {
        for (const auto& texture : resTextures)
        {
            CUDA_CHECK(cudaDestroyTextureObject(texture.textureObj));
            CUDA_CHECK(cudaFreeArray(texture.pixelArray));
        }
    }
}

void NodeEvaluator::setOutputNode(Node* outputNode)
{
    this->outputNode = outputNode;
}

Texture NodeEvaluator::requestTexture(glm::ivec2 resolution)
{
    if (this->textures.contains(resolution))
    {
        for (const auto& texture : this->textures[resolution])
        {
            if (texture.numReferences == 0)
            {
                return texture;
            }
        }
    }
    else
    {
        this->textures[resolution] = std::vector<Texture>();
    }

    Texture tex;
    tex.resolution = resolution;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    int pitch = resolution.x * 4 * sizeof(float);

    CUDA_CHECK(cudaMallocArray(&tex.pixelArray, &channelDesc, resolution.x, resolution.y));

    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = tex.pixelArray;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeWrap;
    tex_desc.addressMode[1] = cudaAddressModeWrap;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    tex_desc.maxAnisotropy = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode = cudaFilterModeLinear;
    tex_desc.borderColor[0] = 1.0f;
    tex_desc.sRGB = 0;

    CUDA_CHECK(cudaCreateTextureObject(&tex.textureObj, &res_desc, &tex_desc, nullptr));

    this->textures[resolution].push_back(tex);
    return tex;
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
        node->evaluate();
        node->clearInputTextures();
        printf("%s\n", node->name.c_str());
    }
    printf("\n");
}