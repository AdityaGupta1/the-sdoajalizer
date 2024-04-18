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

protected:
    unsigned int getTitleBarColor() const override;
    unsigned int getTitleBarSelectedColor() const override;

private:
    bool isFileExr() const;

    void reloadFile();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void _evaluate() override;
};
