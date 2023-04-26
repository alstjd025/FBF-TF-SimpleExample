#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace std;

int main()
{
    const char* socket_path = "/tmp/my_socket";
    char buffer[1024];
    struct sockaddr_un server_address;
    int server_fd, client_fd;

    // Create a Unix domain socket
    server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd == -1) {
        cerr << "Failed to create a socket\n";
        exit(EXIT_FAILURE);
    }

    // Set the socket address
    memset(&server_address, 0, sizeof(struct sockaddr_un));
    server_address.sun_family = AF_UNIX;
    strncpy(server_address.sun_path, socket_path, sizeof(server_address.sun_path) - 1);

    // Bind the socket to the address
    if (bind(server_fd, (struct sockaddr*) &server_address, sizeof(struct sockaddr_un)) == -1) {
        cerr << "Failed to bind the socket\n";
        exit(EXIT_FAILURE);
    }

    // Listen for incoming connections
    if (listen(server_fd, 5) == -1) {
        cerr << "Failed to listen for incoming connections\n";
        exit(EXIT_FAILURE);
    }

    // Accept an incoming connection
    client_fd = accept(server_fd, NULL, NULL);
    if (client_fd == -1) {
        cerr << "Failed to accept an incoming connection\n";
        exit(EXIT_FAILURE);
    }

    // Read data from the client
    ssize_t n = read(client_fd, buffer, sizeof(buffer));
    if (n == -1) {
        cerr << "Failed to read data from the client\n";
        exit(EXIT_FAILURE);
    }

    // Print the data received from the client
    cout << "Data received from the client: " << buffer << endl;

    // Write data back to the client
    if (write(client_fd, "Hello from server", 17) == -1) {
        cerr << "Failed to write data back to the client\n";
        exit(EXIT_FAILURE);
    }

    // Close the sockets
    close(client_fd);
    close(server_fd);

    return 0;
}