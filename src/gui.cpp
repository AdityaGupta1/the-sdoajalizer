#include "gui.hpp"

void Gui::init(GLFWwindow* window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    io = &ImGui::GetIO();
    ImGui::StyleColorsLight();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");
}

void Gui::deinit()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Gui::render()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();

	ImGui::Begin("Path Tracer Analytics");

	ImGui::Text("joe biden");

	//ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
	//ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
	//ImGui::Checkbox("Another Window", &show_another_window);

	//ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
	//ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

	//if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
	//	counter++;
	//ImGui::SameLine();
	//ImGui::Text("counter = %d", counter);
	//ImGui::Text("Traced Depth %d", imguiData->TracedDepth);
	//ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	ImGui::End();


	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}