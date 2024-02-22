#pragma once

#include "nodes/node.hpp"

class NodeFileInput : public Node
{
private:
    std::string filePath;

    Texture* texFile{ nullptr };
    bool needsReloadFile{ false };

    static std::vector<const char*> colorSpaceOptions;
    int selectedColorSpace{ 0 }; // linear

public:
    NodeFileInput();

private:
    void reloadFile();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;

    std::string debugGetSrcFileName() const override;
};
