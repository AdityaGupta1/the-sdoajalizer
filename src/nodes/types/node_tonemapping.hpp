#pragma once

#include "nodes/node.hpp"

class NodeToneMapping : public Node
{
private:
    static std::vector<const char*> toneMappingOptions;
    int selectedToneMapping{ 1 }; // AgX

public:
    NodeToneMapping();

protected:
    unsigned int getTitleBarColor() const override;
    unsigned int getTitleBarSelectedColor() const override;

    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void _evaluate() override;
};
