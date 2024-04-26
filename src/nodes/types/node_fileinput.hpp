#pragma once

#include "nodes/node.hpp"

class NodeFileInput : public Node
{
private:
    std::string filePath;

    static std::vector<const char*> colorSpaceOptions;
    int selectedColorSpace{ 0 }; // linear

public:
    NodeFileInput();

protected:
    unsigned int getTitleBarColor() const override;
    unsigned int getTitleBarHoveredColor() const override;

    bool drawPinExtras(const Pin* pin, int pinNumber) override;

private:
    bool isFileExr() const;

protected:
    void _evaluate() override;
};
