#pragma once

#include "nodes/node.hpp"

class NodeFileInput : public Node
{
private:
    std::string filePath{ "" };

    Texture* texFile{ nullptr };
    bool needsReloadFile{ false };

public:
    NodeFileInput();

private:
    void reloadFile();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;
};
