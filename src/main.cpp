#include "main.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include "ImGui/imgui.h"
#include "ImGui/imgui_impl_glfw.h"
#include "ImGui/imgui_impl_opengl3.h"

GLFWwindow* window;

int width = 960;
int height = 540;

void errorCallback(int error, const char* description)
{
    fprintf(stderr, "%s\n", description);
}

int main(int argc, char* argv[]) 
{
    std::cout << "welcome to sdoaj's america" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count)
    {
        std::cerr << "error: GPU device number greater than device count" << std::endl;
        return -1;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::cout << "device name:         " << deviceProp.name << std::endl;
    std::cout << "compute capability:  " << major << "." << minor << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    glfwSetErrorCallback(errorCallback);

    if (!glfwInit())
    {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "the sdoajalizer", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        return -1;
    }
    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    std::cout << "------------------------------------------------------------" << std::endl;

    mainLoop();

    return 0;
}

void mainLoop()
{
    while (!glfwWindowShouldClose(window))
    {

        glfwPollEvents();

        // TODO

        glfwSwapBuffers(window);
    }

    //ImGui_ImplOpenGL3_Shutdown();
    //ImGui_ImplGlfw_Shutdown();
    //ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
        case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GL_TRUE);
            break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    // TODO
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    // TODO
}
