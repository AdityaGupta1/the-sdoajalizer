#include "node_evaluator.hpp"

#include <queue>
#include <unordered_map>

#include <iostream>

NodeEvaluator::NodeEvaluator()
{}

void NodeEvaluator::setOutputNode(Node* outputNode)
{
    this->outputNode = outputNode;
}

void NodeEvaluator::evaluate()
{
    std::queue<Node*> nodesWithIndegreeZero;
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

    for (const auto& [node, indegree] : indegrees)
    {
        printf("indegree: %d\n", indegree);
    }
    printf("\n");
}