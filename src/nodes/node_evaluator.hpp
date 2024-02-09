#pragma once

#include "node.hpp"
#include "edge.hpp"
#include "../texture.hpp"

#include <unordered_map>
#include <memory>

#include <glm/glm.hpp>
#include <cuda_runtime.h>

class Node;
class Pin;
class Edge;

class NodeEvaluator
{
private:
    Node* outputNode{ nullptr };

    std::unordered_map<glm::ivec2, std::vector<std::unique_ptr<Texture>>, ResolutionHash> textures;

public:
    const glm::ivec2 outputResolution;

    NodeEvaluator(glm::ivec2 outputResolution);
    ~NodeEvaluator();

    void setOutputNode(Node* outputNode);

    Texture* requestTexture(); // defaults to output resolution
    Texture* requestTexture(glm::ivec2 resolution);

    void evaluate();
};