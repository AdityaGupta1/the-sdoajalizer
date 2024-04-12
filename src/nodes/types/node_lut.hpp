#pragma once

#include "nodes/node.hpp"

class NodeLUT : public Node
{
private:
    std::string filePath;

    cudaArray_t lutArray{ nullptr };
    cudaTextureObject_t lutTexObj;
    bool needsReloadFile{ false };

public:
    NodeLUT();
    ~NodeLUT() override;

private:
    void freeTextureArray();

    void reloadFile();

protected:
    bool drawPinExtras(const Pin* pin, int pinNumber) override;
    void evaluate() override;
};
