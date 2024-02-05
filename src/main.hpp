#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

int main(int argc, char* argv[]);

void mainLoop();

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
