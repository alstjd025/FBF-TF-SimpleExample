#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/un.h> 
#include <iostream>
#include <pack.h>

#define  BUFF_SIZE   1024
#define  SOCK_TARGETFILE  "tmp/process_a"
#define SOCK_LOCALFILE "tmp/client"

int   main( int argc, char **argv)
{
   int    sock;
   size_t addr_size;
   struct sockaddr_un   local_addr;
   struct sockaddr_un   target_addr;
   char   buff_rcv[BUFF_SIZE];
   if(argc < 2){
      std::cout << "Need sock name for arg" << "\n";
   }
   if ( 0 == access( SOCK_LOCALFILE, F_OK))
      unlink( SOCK_LOCALFILE);


   sock  = socket(PF_FILE, SOCK_DGRAM, 0);
   if(sock == -1){
      std::cout << "socker error" << "\n";
   }
   
   if( -1 == sock)
   {
      printf( "socket create ERROR \n");
      exit( 1);
   }
   memset(&target_addr, 0, sizeof(target_addr));
   target_addr.sun_family = AF_UNIX;
   strcpy(target_addr.sun_path, SOCK_TARGETFILE);
   addr_size = sizeof(target_addr);

   memset(&local_addr, 0, sizeof(local_addr));
   local_addr.sun_family = AF_UNIX;
   strcpy(local_addr.sun_path, SOCK_LOCALFILE);

   if(bind(sock, (struct sockaddr*)&local_addr, sizeof(local_addr)) == -1){
      std::cout << "bind error" << "\n";
      exit(1);
   }

   if(sendto(sock, "asdfasdf", BUFF_SIZE, 0, 
            (struct sockaddr*)&target_addr, sizeof(target_addr)) == -1){
      std::cout << "send ERROR" << "\n";
   }

   if(recvfrom(sock, buff_rcv, BUFF_SIZE, 0 , 
                 NULL, 0) == -1){
   std::cout << "ERROR \n";  
  }

   printf("%s \n", buff_rcv);
   close(sock);
   
   return 0;
}