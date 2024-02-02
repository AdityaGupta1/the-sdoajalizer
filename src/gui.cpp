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

	// https://gist.github.com/AidanSun05/953f1048ffe5699800d2c92b88c36d9f
	if (firstRender)
	{
		ImVec2 workCenter = ImGui::GetMainViewport()->GetWorkCenter();

		ImGuiID id = ImGui::GetID("MainWindowGroup");
		ImGui::DockBuilderRemoveNode(id);
		ImGui::DockBuilderAddNode(id);

		ImVec2 nodeSize{ 960, 540 }; // TODO: support resizing?
		ImVec2 nodePos{ workCenter.x - nodeSize.x * 0.5f, workCenter.y - nodeSize.y * 0.5f };

		ImGui::DockBuilderSetNodeSize(id, nodeSize);
		ImGui::DockBuilderSetNodePos(id, nodePos);

		ImGuiID dock1 = ImGui::DockBuilderSplitNode(id, ImGuiDir_Left, 0.5f, nullptr, &id);
		ImGuiID dock2 = ImGui::DockBuilderSplitNode(id, ImGuiDir_Right, 0.5f, nullptr, &id);
		ImGuiID dock3 = ImGui::DockBuilderSplitNode(dock2, ImGuiDir_Down, 0.5f, nullptr, &dock2);

		ImGui::DockBuilderDockWindow("One", dock1);
		ImGui::DockBuilderDockWindow("Two", dock2);
		ImGui::DockBuilderDockWindow("Three", dock3);

		ImGui::DockBuilderFinish(id);
	}

	ImGui::Begin("One");

	ImGui::Text("The FitnessGram™ Pacer Test is a multistage aerobic capacity test that progressively gets more difficult as it continues. The 20 meter pacer test will begin in 30 seconds. Line up at the start. The running speed starts slowly, but gets faster each minute after you hear this signal. [beep] A single lap should be completed each time you hear this sound. [ding] Remember to run in a straight line, and run as long as possible. The second time you fail to complete a lap before the sound, your test is over. The test will begin on the word start. On your mark, get ready, start.");

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

	ImGui::Begin("Two");

	ImGui::Text("Hello, I am currently 16 years old and I want to become a walrus. I know there’s a million people out there just like me, but I promise you I’m different. On December 14th, I’m moving to Antarctica; home of the greatest walruses. I’ve already cut off my arms, and now slide on my stomach everywhere I go as training. I may not be a walrus yet, but I promise you if you give me a chance and the support I need, I will become the greatest walrus ever.");

	ImGui::End();

	ImGui::Begin("Three");

	ImGui::Text("Hello, I am currently 16 years old and I want to become a walrus. I know there’s a million people out there just like me, but I promise you I’m different. On December 14th, I’m moving to Antarctica; home of the greatest walruses. I’ve already cut off my arms, and now slide on my stomach everywhere I go as training. I may not be a walrus yet, but I promise you if you give me a chance and the support I need, I will become the greatest walrus ever.");

	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	if (firstRender)
	{
		firstRender = false;
	}
}
