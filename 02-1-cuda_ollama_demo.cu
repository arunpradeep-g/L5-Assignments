// CUDA device check + local Ollama LLM prompt for Windows.
// Build with: nvcc cuda_ollama_demo.cu -o cuda_ollama_demo.exe -Xcompiler /EHsc
#include <cuda_runtime.h>
#include <windows.h>
#include <winhttp.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#pragma comment(lib, "winhttp.lib")

__global__ void add_one(int* value) { *value += 1; }

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

std::string json_escape(const std::string& text) {
    std::string escaped;
    for (char c : text) {
        switch (c) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += c;
        }
    }
    return escaped;
}

std::string ollama_generate(const std::string& prompt) {
    const std::string request = "{\"model\":\"llama3.2\",\"prompt\":\"" + json_escape(prompt) +
                                "\",\"stream\":false,\"options\":{\"temperature\":0.2}}";
    HINTERNET session = WinHttpOpen(L"CUDA-Ollama-Demo/1.0", WINHTTP_ACCESS_TYPE_NO_PROXY,
                                   WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!session) throw std::runtime_error("Unable to open a WinHTTP session.");
    HINTERNET connection = WinHttpConnect(session, L"127.0.0.1", 11434, 0);
    HINTERNET request_handle = connection ? WinHttpOpenRequest(connection, L"POST", L"/api/generate", nullptr,
        WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0) : nullptr;
    const wchar_t* headers = L"Content-Type: application/json\r\n";
    BOOL sent = request_handle && WinHttpSendRequest(request_handle, headers, -1L,
        const_cast<char*>(request.data()), static_cast<DWORD>(request.size()), static_cast<DWORD>(request.size()), 0) &&
        WinHttpReceiveResponse(request_handle, nullptr);
    if (!sent) {
        if (request_handle) WinHttpCloseHandle(request_handle);
        if (connection) WinHttpCloseHandle(connection);
        WinHttpCloseHandle(session);
        throw std::runtime_error("Could not contact Ollama at http://127.0.0.1:11434. Start Ollama first.");
    }
    std::string response;
    DWORD available = 0;
    while (WinHttpQueryDataAvailable(request_handle, &available) && available) {
        std::vector<char> buffer(available + 1);
        DWORD received = 0;
        if (!WinHttpReadData(request_handle, buffer.data(), available, &received)) break;
        response.append(buffer.data(), received);
    }
    WinHttpCloseHandle(request_handle); WinHttpCloseHandle(connection); WinHttpCloseHandle(session);
    return response;
}

int main(int argc, char** argv) {
    try {
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) throw std::runtime_error("No CUDA-capable NVIDIA GPU was found.");
        cudaDeviceProp gpu{};
        check_cuda(cudaGetDeviceProperties(&gpu, 0), "cudaGetDeviceProperties");
        std::cout << "CUDA device: " << gpu.name << " (compute capability " << gpu.major << "." << gpu.minor << ")\n";
        int host_value = 41, *device_value = nullptr;
        check_cuda(cudaMalloc(&device_value, sizeof(int)), "cudaMalloc");
        check_cuda(cudaMemcpy(device_value, &host_value, sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
        add_one<<<1, 1>>>(device_value);
        check_cuda(cudaGetLastError(), "CUDA kernel launch");
        check_cuda(cudaMemcpy(&host_value, device_value, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
        cudaFree(device_value);
        std::cout << "CUDA kernel check: 41 + 1 = " << host_value << "\n\n";
        std::string prompt = argc > 1 ? argv[1] : "In one sentence, explain why GPU acceleration is useful for local LLM inference.";
        //std::cout << "Prompt: " << prompt << "\n\nOllama response JSON:\n" << ollama_generate(prompt) << "\n";
		std::string raw = ollama_generate(prompt);

		// Find the "response" field in the JSON
		std::string key = "\"response\":\"";
		size_t start = raw.find(key);
		std::string answer;

		if (start != std::string::npos) {
			start += key.length();
			size_t end = raw.find("\"", start);
			if (end != std::string::npos) {
				answer = raw.substr(start, end - start);
			}
		}

		// Unescape common sequences (optional, minimal)
		for (size_t pos = 0; (pos = answer.find("\\n", pos)) != std::string::npos; )
			answer.replace(pos, 2, "\n");
		for (size_t pos = 0; (pos = answer.find("\\\"", pos)) != std::string::npos; )
			answer.replace(pos, 2, "\"");

		std::cout << "Prompt: " << prompt << "\n\nOllama response:\n" << answer << "\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
