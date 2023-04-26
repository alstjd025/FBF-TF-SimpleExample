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
#define  SOCK_LOCALFILE   "tmp/process_a"

int   main( void)
{
   int    sock;
   size_t addr_size;
   struct sockaddr_un   local_addr;
   struct sockaddr_un   guest_addr;
   char   buff_rcv[BUFF_SIZE];
   char   buff_snd[BUFF_SIZE];

   if ( 0 == access( SOCK_LOCALFILE, F_OK))
      unlink( SOCK_LOCALFILE);

   sock  = socket(PF_FILE, SOCK_DGRAM, 0);

   if( -1 == sock)
   {
      printf( "socket create ERROR \n");
      exit( 1);
   }

   memset(&local_addr, 0, sizeof( local_addr));
   local_addr.sun_family = AF_UNIX;
   strcpy(local_addr.sun_path, SOCK_LOCALFILE);

   if( -1 == bind(sock, (struct sockaddr*)&local_addr, sizeof(local_addr)) )
   {
      printf( "bind() ERROR \n");
      exit( 1);
   }
   while(1)
   {
      addr_size  = sizeof(guest_addr);
      if(recvfrom(sock, buff_rcv, BUFF_SIZE, 0 ,
                     (struct sockaddr*)&guest_addr, (socklen_t*)&addr_size) == -1){
        std::cout << "ERROR with recv" << "\n";
      }
      printf("receive: %s from %s\n", buff_rcv, guest_addr.sun_path);

      strcpy(buff_snd, "go to hell");
      if(sendto(sock, buff_snd, BUFF_SIZE, 0,
                     (struct sockaddr*)&guest_addr, sizeof(guest_addr)) == -1){
        std::cout << "ERROR \n";
      }
   }
}