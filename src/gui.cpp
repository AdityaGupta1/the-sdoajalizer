#include "gui.hpp"

#include <GLFW/glfw3.h>

#include "ImGui/imgui_internal.h"
#include "ImGui/imnodes.h"
#include "ImGui/misc/cpp/imgui_stdlib.h"
#include "ImGui/combo_filter/imgui_combo_filter.h"

#include <iostream>

#include "nodes/all_nodes.hpp"

void Gui::init(GLFWwindow* window)
{
    this->window = window;

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImNodes::CreateContext();
    io = &ImGui::GetIO();
    setupStyle();

    ImFont* font = io->Fonts->AddFontFromFileTTF("fonts/Inter-Regular.ttf", 16);

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io->ConfigDockingWithShift = false;
    io->ConfigDragClickToInputText = true;

    nodeEvaluator.init();

    auto outputNodeUptr = std::make_unique<NodeOutput>();
    this->outputNode = outputNodeUptr.get();
    addNode(std::move(outputNodeUptr));

    this->nodeEvaluator.setOutputNode(this->outputNode);
}

void Gui::setupStyle()
{
    // Soft Cherry style by Patitotective from ImThemes
    ImGuiStyle& style = ImGui::GetStyle();

    style.Alpha = 1.0f;
    style.DisabledAlpha = 0.4000000059604645f;
    style.WindowPadding = ImVec2(3.0f, 3.0f);
    style.WindowRounding = 0.0f;
    style.WindowBorderSize = 0.0f;
    style.WindowMinSize = ImVec2(50.0f, 50.0f);
    style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
    style.WindowMenuButtonPosition = ImGuiDir_Left;
    style.ChildRounding = 0.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupRounding = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FramePadding = ImVec2(5.0f, 3.0f);
    style.FrameRounding = 3.0f;
    style.FrameBorderSize = 0.0f;
    style.ItemSpacing = ImVec2(6.0f, 6.0f);
    style.ItemInnerSpacing = ImVec2(3.0f, 2.0f);
    style.CellPadding = ImVec2(3.0f, 3.0f);
    style.IndentSpacing = 6.0f;
    style.ColumnsMinSpacing = 6.0f;
    style.ScrollbarSize = 13.0f;
    style.ScrollbarRounding = 16.0f;
    style.GrabMinSize = 20.0f;
    style.GrabRounding = 4.0f;
    style.TabRounding = 4.0f;
    style.TabBorderSize = 1.0f;
    style.TabMinWidthForCloseButton = 0.0f;
    style.ColorButtonPosition = ImGuiDir_Right;
    style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
    style.SelectableTextAlign = ImVec2(0.0f, 0.0f);

    style.Colors[ImGuiCol_Text] = ImVec4(0.8588235378265381f, 0.929411768913269f, 0.886274516582489f, 1.0f);
    style.Colors[ImGuiCol_TextDisabled] = ImVec4(0.5215686559677124f, 0.5490196347236633f, 0.5333333611488342f, 1.0f);
    style.Colors[ImGuiCol_WindowBg] = ImVec4(0.1294117718935013f, 0.1372549086809158f, 0.168627455830574f, 1.0f);
    style.Colors[ImGuiCol_ChildBg] = ImVec4(0.1490196138620377f, 0.1568627506494522f, 0.1882352977991104f, 1.0f);
    style.Colors[ImGuiCol_PopupBg] = ImVec4(0.2000000029802322f, 0.2196078449487686f, 0.2666666805744171f, 1.0f);
    style.Colors[ImGuiCol_Border] = ImVec4(0.1372549086809158f, 0.1137254908680916f, 0.1333333402872086f, 1.0f);
    style.Colors[ImGuiCol_BorderShadow] = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_FrameBg] = ImVec4(0.168627455830574f, 0.1843137294054031f, 0.2313725501298904f, 1.0f);
    style.Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.4549019634723663f, 0.196078434586525f, 0.2980392277240753f, 1.0f);
    style.Colors[ImGuiCol_FrameBgActive] = ImVec4(0.4549019634723663f, 0.196078434586525f, 0.2980392277240753f, 1.0f);
    style.Colors[ImGuiCol_TitleBg] = ImVec4(0.2313725501298904f, 0.2000000029802322f, 0.2705882489681244f, 1.0f);
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4(0.501960813999176f, 0.07450980693101883f, 0.2549019753932953f, 1.0f);
    style.Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.2000000029802322f, 0.2196078449487686f, 0.2666666805744171f, 1.0f);
    style.Colors[ImGuiCol_MenuBarBg] = ImVec4(0.2000000029802322f, 0.2196078449487686f, 0.2666666805744171f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.239215686917305f, 0.239215686917305f, 0.2196078449487686f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.3882353007793427f, 0.3882353007793427f, 0.3725490272045135f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.6941176652908325f, 0.6941176652908325f, 0.686274528503418f, 1.0f);
    style.Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.6941176652908325f, 0.6941176652908325f, 0.686274528503418f, 1.0f);
    style.Colors[ImGuiCol_CheckMark] = ImVec4(0.658823549747467f, 0.1372549086809158f, 0.1764705926179886f, 1.0f);
    style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.6509804129600525f, 0.1490196138620377f, 0.3450980484485626f, 1.0f);
    style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.7098039388656616f, 0.2196078449487686f, 0.2666666805744171f, 1.0f);
    style.Colors[ImGuiCol_Button] = ImVec4(0.6509804129600525f, 0.1490196138620377f, 0.3450980484485626f, 1.0f);
    style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.4549019634723663f, 0.196078434586525f, 0.2980392277240753f, 1.0f);
    style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.4549019634723663f, 0.196078434586525f, 0.2980392277240753f, 1.0f);
    style.Colors[ImGuiCol_Header] = ImVec4(0.4549019634723663f, 0.196078434586525f, 0.2980392277240753f, 1.0f);
    style.Colors[ImGuiCol_HeaderHovered] = ImVec4(0.6509804129600525f, 0.1490196138620377f, 0.3450980484485626f, 1.0f);
    style.Colors[ImGuiCol_HeaderActive] = ImVec4(0.501960813999176f, 0.07450980693101883f, 0.2549019753932953f, 1.0f);
    style.Colors[ImGuiCol_Separator] = ImVec4(0.4274509847164154f, 0.4274509847164154f, 0.4980392158031464f, 1.0f);
    style.Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.09803921729326248f, 0.4000000059604645f, 0.7490196228027344f, 1.0f);
    style.Colors[ImGuiCol_SeparatorActive] = ImVec4(0.09803921729326248f, 0.4000000059604645f, 0.7490196228027344f, 1.0f);
    style.Colors[ImGuiCol_ResizeGrip] = ImVec4(0.6509804129600525f, 0.1490196138620377f, 0.3450980484485626f, 1.0f);
    style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.4549019634723663f, 0.196078434586525f, 0.2980392277240753f, 1.0f);
    style.Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.4549019634723663f, 0.196078434586525f, 0.2980392277240753f, 1.0f);
    style.Colors[ImGuiCol_Tab] = ImVec4(0.1764705926179886f, 0.3490196168422699f, 0.5764706134796143f, 1.0f);
    style.Colors[ImGuiCol_TabHovered] = ImVec4(0.2588235437870026f, 0.5882353186607361f, 0.9764705896377563f, 1.0f);
    style.Colors[ImGuiCol_TabActive] = ImVec4(0.196078434586525f, 0.407843142747879f, 0.6784313917160034f, 1.0f);
    style.Colors[ImGuiCol_TabUnfocused] = ImVec4(0.06666667014360428f, 0.1019607856869698f, 0.1450980454683304f, 1.0f);
    style.Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.1333333402872086f, 0.2588235437870026f, 0.4235294163227081f, 1.0f);
    style.Colors[ImGuiCol_PlotLines] = ImVec4(0.8588235378265381f, 0.929411768913269f, 0.886274516582489f, 1.0f);
    style.Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.4549019634723663f, 0.196078434586525f, 0.2980392277240753f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogram] = ImVec4(0.3098039329051971f, 0.7764706015586853f, 0.196078434586525f, 1.0f);
    style.Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.4549019634723663f, 0.196078434586525f, 0.2980392277240753f, 1.0f);
    style.Colors[ImGuiCol_TableHeaderBg] = ImVec4(0.1882352977991104f, 0.1882352977991104f, 0.2000000029802322f, 1.0f);
    style.Colors[ImGuiCol_TableBorderStrong] = ImVec4(0.3098039329051971f, 0.3098039329051971f, 0.3490196168422699f, 1.0f);
    style.Colors[ImGuiCol_TableBorderLight] = ImVec4(0.2274509817361832f, 0.2274509817361832f, 0.2470588237047195f, 1.0f);
    style.Colors[ImGuiCol_TableRowBg] = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_TableRowBgAlt] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.3843137323856354f, 0.6274510025978088f, 0.9176470637321472f, 1.0f);
    style.Colors[ImGuiCol_DragDropTarget] = ImVec4(1.0f, 1.0f, 0.0f, 1.0f);
    style.Colors[ImGuiCol_NavHighlight] = ImVec4(0.2588235437870026f, 0.5882353186607361f, 0.9764705896377563f, 1.0f);
    style.Colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
    style.Colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.800000011920929f, 0.800000011920929f, 0.800000011920929f, 1.0f);
    style.Colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.800000011920929f, 0.800000011920929f, 0.800000011920929f, 0.300000011920929f);
}

void Gui::deinit()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImNodes::DestroyContext();
    ImGui::DestroyContext();
}

Pin& Gui::getPin(int pinId)
{
    return this->nodes[pinId - (pinId % NODE_ID_STRIDE)]->getPin(pinId);
}

void Gui::addNode(std::unique_ptr<Node> node)
{
    node->setNodeEvaluator(&this->nodeEvaluator);
    this->nodes[node->id] = std::move(node);
}

void Gui::addEdge(int startPinId, int endPinId)
{
    Pin& startPin = getPin(startPinId);
    Pin& endPin = getPin(endPinId);

    deletePinEdges(endPin);

    auto edgePtr = std::make_unique<Edge>(&startPin, &endPin);
    startPin.addEdge(edgePtr.get());
    endPin.addEdge(edgePtr.get());
    this->edges[edgePtr->id] = std::move(edgePtr);

    this->isNetworkDirty = true;
}

void Gui::deleteNode(int nodeId)
{
    if (nodeId == 0) // prevent deletion of output node
    {
        return;
    }

    const auto& node = this->nodes[nodeId];

    for (auto& inputPin : node->inputPins)
    {
        deletePinEdges(inputPin);
    }

    for (auto& outputPin : node->outputPins)
    {
        deletePinEdges(outputPin);
    }

    this->nodes.erase(nodeId);
}

void Gui::deleteEdge(int edgeId)
{
    const auto edge = this->edges[edgeId].get();
    edge->startPin->removeEdge(edge);
    edge->endPin->removeEdge(edge);
    this->edges.erase(edgeId);
}

void Gui::deleteEdge(Edge* edge)
{
    edge->startPin->removeEdge(edge);
    edge->endPin->removeEdge(edge);
    this->edges.erase(edge->id);
}

void Gui::deletePinEdges(Pin& pin)
{
    std::vector<int> edgesToDelete;

    for (const auto& edge : pin.getEdges())
    {
        edgesToDelete.push_back(edge->id);
    }

    for (int edgeId : edgesToDelete)
    {
        deleteEdge(edgeId);
    }

    pin.clearEdges();
}

void Gui::render()
{
    if (isNetworkDirty)
    {
        isNetworkDirty = false;

        nodeEvaluator.evaluate();
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetMainViewport()->WorkSize);

    ImGui::Begin("DockSpace", nullptr, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("New", "Ctrl+N")) // this only displays the shortcut text, doesn't actually make it work
            {
                printf("New\n");
            }

            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }

    ImGuiDockNodeFlags dockSpaceFlags = ImGuiDockNodeFlags_PassthruCentralNode;

    ImGuiID dockspace_id = ImGui::GetID("MyDockspace");
    ImGui::DockSpace(dockspace_id, ImVec2(0, 0), dockSpaceFlags);

    if (isFirstRender)
    {
        ImGui::DockBuilderRemoveNode(dockspace_id);
        ImGui::DockBuilderAddNode(dockspace_id, dockSpaceFlags | ImGuiDockNodeFlags_DockSpace);

        ImGuiID viewer = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Left, 0.5f, nullptr, &dockspace_id);

        ImGui::DockBuilderDockWindow("Viewer", viewer);
        ImGui::DockBuilderDockWindow("Node Editor", dockspace_id);

        ImGui::DockBuilderFinish(dockspace_id);
    }

    ImGui::End();

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus;

    // VIEWER
    // ================================================================================

    ImGui::Begin("Viewer", nullptr, windowFlags);
    drawOutputImageViewer();
    ImGui::End();

    // NODE EDITOR
    // ================================================================================

    ImGui::Begin("Node Editor", nullptr, windowFlags);

    if (isFirstRender)
    {
        ImVec2 windowSize = ImGui::GetWindowSize();
        ImNodes::SetNodeEditorSpacePos(0, ImVec2(windowSize.x * 0.8f, windowSize.y * 0.5f)); // 0 = output node
    }

    drawNodeEditor();
    ImGui::End();

    // ================================================================================

    updateNodeCreatorWindow();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (isFirstRender)
    {
        isFirstRender = false;
    }
}

void Gui::drawOutputImageViewer()
{
    if (!nodeEvaluator.hasOutputTexture())
    {
        return;
    }

    ImVec2 contentSize = ImGui::GetContentRegionAvail();
    float contentAspectRatio = contentSize.y / contentSize.x;

    float imageAspectRatio = nodeEvaluator.outputResolution.y / (float)nodeEvaluator.outputResolution.x;

    ImVec2 imageSize;
    if (contentAspectRatio < imageAspectRatio)
    {
        imageSize.y = contentSize.y;
        imageSize.x = imageSize.y / imageAspectRatio;
    }
    else
    {
        imageSize.x = contentSize.x;
        imageSize.y = imageSize.x * imageAspectRatio;
    }

    ImVec2 oldCursorPos = ImGui::GetCursorScreenPos();
    ImVec2 newCursorPos = (contentSize - imageSize) * 0.5f;
    newCursorPos.y += oldCursorPos.y;
    ImGui::SetCursorScreenPos(newCursorPos);

    ImGui::Image((void*)(intptr_t)nodeEvaluator.viewerTex, imageSize);
}

void Gui::drawNodeEditor()
{
    ImNodes::BeginNodeEditor();

    for (const auto& [nodeId, node] : nodes)
    {
        bool wasParameterChanged = node->draw();
        if (wasParameterChanged)
        {
            isNetworkDirty = true;
        }
    }

    for (const auto& [edgeId, edge] : edges)
    {
        ImNodes::Link(edgeId, edge->startPin->id, edge->endPin->id);
    }

    ImNodes::EndNodeEditor();

    int startPinId, endPinId;
    if (ImNodes::IsLinkCreated(&startPinId, &endPinId))
    {
        addEdge(startPinId, endPinId);
    }

    const int numSelectedNodes = ImNodes::NumSelectedNodes();
    std::vector<int> selectedNodes;
    if (numSelectedNodes > 0)
    {
        selectedNodes.resize(numSelectedNodes);
        ImNodes::GetSelectedNodes(selectedNodes.data());
    }

    if (controls.deleteComponents)
    {
        bool didDelete = false;

        const int numSelectedEdges = ImNodes::NumSelectedLinks();
        if (numSelectedEdges > 0)
        {
            std::vector<int> selectedEdges;
            selectedEdges.resize(numSelectedEdges);
            ImNodes::GetSelectedLinks(selectedEdges.data());
            for (const auto edgeId : selectedEdges)
            {
                deleteEdge(edgeId);
                didDelete = true;
            }
        }

        for (const auto nodeId : selectedNodes)
        {
            deleteNode(nodeId);
            didDelete = true;
        }

        if (didDelete)
        {
            isNetworkDirty = true; // TODO: check if the deleted objects were actually connected to the output
        }
    }

    if (controls.debugOpenSrcFile && numSelectedNodes == 1) 
    {
        controls.debugOpenSrcFile = false;
        this->nodes[selectedNodes[0]]->debugOpenSrcFile();
    }

    controls.deleteComponents = false;
    controls.debugOpenSrcFile = false;
}

const char* itemGetter(const std::vector<std::pair<std::string, std::function<std::unique_ptr<Node>()>>>& items, int index) {
    if (index >= 0 && index < (int)items.size()) {
        return items[index].first.c_str();
    }
    return "N/A";
}

static bool fuzzyScore(const char* str1, const char* str2, int& score)
{
    score = 0;
    if (*str2 == '\0')
    {
        return *str1 == '\0';
    }

    int consecutive = 0;
    int maxerrors = 0;

    while (*str1 && *str2) {
        int is_leading = (*str1 & 64) && !(str1[1] & 64);
        if ((*str1 & ~32) == (*str2 & ~32)) {
            int had_separator = (str1[-1] <= 32);
            int x = had_separator || is_leading ? 10 : consecutive * 5;
            consecutive = 1;
            score += x;
            ++str2;
        }
        else {
            int x = -1, y = is_leading * -3;
            consecutive = 0;
            score += x;
            maxerrors += y;
        }
        ++str1;
    }

    score += (maxerrors < -9 ? -9 : maxerrors);
    return *str2 == '\0';
};

template<typename T>
void filterSearch(const ImGui::ComboFilterSearchCallbackData<T>& cbd)
{
    const int items_count = static_cast<int>(std::size(cbd.Items));
    for (int i = 0; i < items_count; ++i) {
        int score = 0;
        if (fuzzyScore(cbd.ItemGetter(cbd.Items, i), cbd.SearchString, score))
            cbd.FilterResults->emplace_back(i, score);
    }

    ImGui::SortFilterResultsDescending(*cbd.FilterResults);
}

static std::vector<std::pair<std::string, std::function<std::unique_ptr<Node>()>>> nodeCreators = {
    { "color", std::make_unique<NodeColor> },
    { "file input", std::make_unique<NodeFileInput> },
    { "invert", std::make_unique<NodeInvert> },
    { "mix", std::make_unique<NodeMix> },
    { "noise", std::make_unique<NodeNoise> },
    { "uv gradient", std::make_unique<NodeUvGradient> },
    { "exposure", std::make_unique<NodeExposure> },
    { "brightness/contrast", std::make_unique<NodeBrightnessContrast> }
};

void Gui::updateNodeCreatorWindow()
{
    if (createWindowData.visible) {
        ImGui::SetNextWindowPos(createWindowData.pos);

        int windowFlags =
            ImGuiWindowFlags_NoDocking |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_AlwaysAutoResize |
            ImGuiWindowFlags_NoTitleBar;
        ImGui::Begin("node creator", nullptr, windowFlags);

        ImGui::PushItemWidth(400);

        int selectedItem = -1;
        if (ImGui::ComboFilter("##nodeSearch", selectedItem, nodeCreators, itemGetter, filterSearch, createWindowData.justOpened, ImGuiComboFlags_NoArrowButton)) {
            auto newNodeUptr = nodeCreators[selectedItem].second();
            int newNodeId = newNodeUptr->id;
            addNode(std::move(newNodeUptr));

            ImNodes::SetNodeScreenSpacePos(newNodeId, ImGui::GetMousePos());

            controls.shouldCreateWindowBeVisible = false;
        }

        ImGui::PopItemWidth();

        ImGui::End();

        createWindowData.justOpened = false;
    }

    if (controls.shouldCreateWindowBeVisible && !createWindowData.visible) {
        createWindowData.visible = true;
        createWindowData.pos = ImGui::GetMousePos();

        createWindowData.justOpened = true;
    }

    if (!controls.shouldCreateWindowBeVisible && createWindowData.visible) {
        createWindowData.visible = false;
    }
}

void Gui::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_BACKSPACE:
            if (io->WantTextInput) {
                break;
            }
            [[fallthrough]];
        case GLFW_KEY_DELETE:
            controls.deleteComponents = true;
            break;
        case GLFW_KEY_TAB:
            if (createWindowData.visible || !io->WantTextInput) {
                controls.shouldCreateWindowBeVisible = !createWindowData.visible;
            }
            break;
        case GLFW_KEY_ESCAPE:
            controls.shouldCreateWindowBeVisible = false;
            break;
        case GLFW_KEY_SLASH:
            controls.debugOpenSrcFile = !io->WantTextInput;
            break;
        }
    }
}

void Gui::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        controls.shouldCreateWindowBeVisible = !createWindowData.visible;
    }
}

void Gui::mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{}
