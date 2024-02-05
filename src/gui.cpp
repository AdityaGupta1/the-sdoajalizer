#include "gui.hpp"

#include "ImGui/imgui_internal.h"

void Gui::init(GLFWwindow* window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO();
    ImGui::StyleColorsLight();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");

    io->ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io->ConfigDockingWithShift = false;
}

void Gui::deinit()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Gui::render()
{
    static bool firstRender = true;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImGui::GetMainViewport()->WorkSize);

    ImGui::Begin("DockSpace", nullptr, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoTitleBar);

    ImGuiDockNodeFlags dockSpaceFlags = ImGuiDockNodeFlags_PassthruCentralNode;

    ImGuiID dockspace_id = ImGui::GetID("MyDockspace");
    ImGui::DockSpace(dockspace_id, ImVec2(0, 0), dockSpaceFlags);

    if (firstRender)
    {
        firstRender = false;

        ImGui::DockBuilderRemoveNode(dockspace_id);
        ImGui::DockBuilderAddNode(dockspace_id, dockSpaceFlags | ImGuiDockNodeFlags_DockSpace);

        ImGuiID viewer1 = ImGui::DockBuilderSplitNode(dockspace_id, ImGuiDir_Up, 0.5f, nullptr, &dockspace_id);
        ImGuiID viewer2 = ImGui::DockBuilderSplitNode(viewer1, ImGuiDir_Right, 0.5f, nullptr, &viewer1);

        ImGui::DockBuilderDockWindow("Viewer 1", viewer1);
        ImGui::DockBuilderDockWindow("Viewer 2", viewer2);
        ImGui::DockBuilderDockWindow("Node Editor", dockspace_id);

        ImGui::DockBuilderFinish(dockspace_id);
    }

    ImGui::End();

    ImGuiWindowFlags windowFlags = ImGuiWindowFlags_NoMove;

    ImGui::Begin("Viewer 1", nullptr, windowFlags);
    ImGui::Text("joe");
    ImGui::End();

    ImGui::Begin("Viewer 2", nullptr, windowFlags);
    ImGui::Text("joe");
    ImGui::End();

    ImGui::Begin("Node Editor", nullptr, windowFlags);
    ImGui::Text("joe");
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (firstRender)
    {
        firstRender = false;
    }
}

//ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
//ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
//ImGui::Checkbox("Another Window", &show_another_window);

//ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
//ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

//if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
//    counter++;
//ImGui::SameLine();
//ImGui::Text("counter = %d", counter);
//ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
//ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
