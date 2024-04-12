#pragma once

#include "nodes/node.hpp"
#include "texture.hpp"

class NodeOutput : public Node
{
private:
    static std::vector<const char*> toneMappingOptions;
    int selectedToneMapping{ 1 }; // AgX

public:
    NodeOutput();

protected:
    unsigned int getTitleBarColor() const override;
    unsigned int getTitleBarSelectedColor() const override;

    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;
};
