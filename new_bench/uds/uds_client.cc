#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <pack.h>

using namespace std;

int main()
{
    const char* socket_path = "tmp/my_socket";
    char buffer[1024];
    struct sockaddr_un server_address;
    int client_fd;

    data_* new_data = new data_;

    // Create a Unix domain socket
    client_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
    if (client_fd == -1) {
        cerr << "Failed to create a socket\n";
        exit(EXIT_FAILURE);
    }

    // Set the socket address
    memset(&server_address, 0, sizeof(struct sockaddr_un));
    server_address.sun_family = AF_UNIX;
    strncpy(server_address.sun_path, socket_path, sizeof(server_address.sun_path) - 1);

    // Connect to the server
    if (connect(client_fd, (struct sockaddr*) &server_address, sizeof(struct sockaddr_un)) == -1) {
        cerr << "Failed to connect to the server\n";
        exit(EXIT_FAILURE);
    }

    // Write data to the server
    if (write(client_fd, "Hello from client", 17) == -1) {
        cerr << "Failed to write data to the server\n";
        exit(EXIT_FAILURE);
    }
    // Read data from the server
    ssize_t n = read(client_fd, buffer, sizeof(buffer));
    if (n == -1) {
        cerr << "Failed to read data from the server\n";
        exit(EXIT_FAILURE);
    }

    // Print the data received from the server
    cout << "Data received from the server: " << buffer << endl;

    // Close the socket
    close(client_fd);

    return 0;
}