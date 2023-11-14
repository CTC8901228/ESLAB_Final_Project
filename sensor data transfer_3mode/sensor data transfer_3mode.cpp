#include "mbed.h"
#include "wifi_helper.h"
#include "mbed-trace/mbed_trace.h"
#include <cstdint>
// #include 

#if MBED_CONF_APP_USE_TLS_SOCKET
#include "root_ca_cert.h"
// #include 
#ifndef DEVICE_TRNG
#error "mbed-os-example-tls-socket requires a device which supports TRNG"
#endif
#endif // MBED_CONF_APP_USE_TLS_SOCKET
#include "stm32l475e_iot01_magneto.h"
#include "stm32l475e_iot01_gyro.h"
#include "stm32l475e_iot01_accelero.h"

Semaphore led_sem(0);
Thread t1;
Thread t2;
Thread t3;
InterruptIn button(BUTTON1);
volatile int button_switch = -1;


class SocketDemo {
    static constexpr size_t MAX_NUMBER_OF_ACCESS_POINTS = 10;
    static constexpr size_t MAX_MESSAGE_RECEIVED_LENGTH = 100;

#if MBED_CONF_APP_USE_TLS_SOCKET
    static constexpr size_t REMOTE_PORT = 42342; // tls port
#else
    static constexpr size_t REMOTE_PORT = 42342; // standard HTTP port
#endif // MBED_CONF_APP_USE_TLS_SOCKET

public:
    SocketDemo() : _net(NetworkInterface::get_default_instance())
    {
    }

    ~SocketDemo()
    {
        if (_net) {
            _net->disconnect();
        }
    }

    void run()
    {
        if (!_net) {
            printf("Error! No network interface found.\r\n");
            return;
        }

        /* if we're using a wifi interface run a quick scan */
        if (_net->wifiInterface()) {
            /* the scan is not required to connect and only serves to show visible access points */
            wifi_scan();
            // 
            /* in this example we use credentials configured at compile time which are used by
             * NetworkInterface::connect() but it's possible to do this at runtime by using the
             * WiFiInterface::connect() which takes these parameters as arguments */
        }

        /* connect will perform the action appropriate to the interface type to connect to the network */

        printf("Connecting to the network...\r\n");
// printf("4324234");
        nsapi_size_or_error_t result = _net->connect();
        if (result != 0) {
            printf("Error! _net->connect() returned: %d\r\n", result);
            return;
        }
// printf("4324234");
        print_network_info();

        /* opening the socket only allocates resources */
        result = _socket.open(_net);
        if (result != 0) {
            printf("Error! _socket.open() returned: %d\r\n", result);
            return;
        }

#if MBED_CONF_APP_USE_TLS_SOCKET
        result = _socket.set_root_ca_cert(root_ca_cert);
        if (result != NSAPI_ERROR_OK) {
            printf("Error: _socket.set_root_ca_cert() returned %d\n", result);
            return;
        }
        _socket.set_hostname(MBED_CONF_APP_HOSTNAME);
#endif // MBED_CONF_APP_USE_TLS_SOCKET

        /* now we have to find where to connect */

        SocketAddress address;

        if (!resolve_hostname(address)) {
            return;
        }

        address.set_port(REMOTE_PORT);

        /* we are connected to the network but since we're using a connection oriented
         * protocol we still need to open a connection on the socket */

        printf("Opening connection to remote port %d\r\n", REMOTE_PORT);

        result = _socket.connect(address);
        if (result != 0) {
            printf("Error! _socket.connect() returned: %d\r\n", result);
            return;
        }
/*
        float sensor_value = 0;
        int16_t pDataXYZ[3] = {0};
        float pGyroDataXYZ[3] = {0};

        printf("Start sensor init\n");
        BSP_ACCELERO_Init();

        while(1) {

            BSP_ACCELERO_AccGetXYZ(pDataXYZ);
            char data[100];
            sprintf(data,"%d %d %d",pDataXYZ[0],pDataXYZ[1],pDataXYZ[2]);
            nsapi_size_t bytes_to_send = strlen(data);
            nsapi_size_or_error_t bytes_sent = 0;

            printf("\r\nSending message: \r\n%s", data);

            while (bytes_to_send) {
                bytes_sent = _socket.send(data + bytes_sent, bytes_to_send);
                printf("sent %d bytes\r\n", bytes_sent);
                bytes_to_send -= bytes_sent;
            }

            ThisThread::sleep_for(1000);

        }
        printf("Demo concluded successfully \r\n");
        */
        return;
    }
/*
    void test(){
        float sensor_value = 0;
        int16_t pDataXYZ[3] = {0};
        float pGyroDataXYZ[3] = {0};

        printf("Start sensor init\n");
        BSP_ACCELERO_Init();

        while(1) {
            BSP_ACCELERO_AccGetXYZ(pDataXYZ);
            char data[100];
            sprintf(data,"%d %d %d",pDataXYZ[0],pDataXYZ[1],pDataXYZ[2]);
            nsapi_size_t bytes_to_send = strlen(data);
            nsapi_size_or_error_t bytes_sent = 0;

            printf("\r\nSending message: \r\n%s", data);

            while (bytes_to_send) {
                bytes_sent = _socket.send(data + bytes_sent, bytes_to_send);
                printf("sent %d bytes\r\n", bytes_sent);
                bytes_to_send -= bytes_sent;
            }

            ThisThread::sleep_for(1000);

        }
        printf("Demo concluded successfully \r\n");
    }
*/
    void thread(){
        while (1) {
            led_sem.acquire();
            while (1) {
                if (button_switch == 1) { 
                    int16_t pACCDataXYZ[3] = {0};
                    int16_t pMAGDataXYZ[3] = {0};
                    float pGyroDataXYZ[3] = {0};
                    printf("Start ACCsensor init\n");
                    BSP_ACCELERO_Init();
                    BSP_MAGNETO_Init();
                    BSP_GYRO_Init();
                    while(button_switch == 1) {
                        BSP_ACCELERO_AccGetXYZ(pACCDataXYZ);
                        BSP_MAGNETO_GetXYZ(pMAGDataXYZ);
                        BSP_GYRO_GetXYZ(pGyroDataXYZ);
                        char data[250];
                    
                        sprintf(data,"%d %d %d %d %d %d %d %d %d %d", button_switch,
                                                                    pACCDataXYZ[0],pACCDataXYZ[1],pACCDataXYZ[2],
                                                                    pMAGDataXYZ[0],pMAGDataXYZ[1],pMAGDataXYZ[2],
                                                                    pGyroDataXYZ[0],pGyroDataXYZ[1],pGyroDataXYZ[2]);
                        nsapi_size_t bytes_to_send = strlen(data);
                        nsapi_size_or_error_t bytes_sent = 0;
                        printf("\r\nSending sensor message: \r\n%s", data);
                        while (bytes_to_send) {
                            bytes_sent = _socket.send(data + bytes_sent, bytes_to_send);
                            printf("sent %d bytes\r\n", bytes_sent);
                            bytes_to_send -= bytes_sent;
                        }
                        ThisThread::sleep_for(1000);
                    }
                    if(button_switch != 1)
                        break;
                }
                if (button_switch == 2) { 
                    int16_t pACCDataXYZ[3] = {0};
                    int16_t pMAGDataXYZ[3] = {0};
                    float pGyroDataXYZ[3] = {0};
                    printf("Start ACCsensor init\n");
                    BSP_ACCELERO_Init();
                    BSP_MAGNETO_Init();
                    BSP_GYRO_Init();
                    while(button_switch == 2) {
                        BSP_ACCELERO_AccGetXYZ(pACCDataXYZ);
                        BSP_MAGNETO_GetXYZ(pMAGDataXYZ);
                        BSP_GYRO_GetXYZ(pGyroDataXYZ);
                        char data[250];
                        sprintf(data,"%d %d %d %d %d %d %d %d %d %d", button_switch,
                                                                    pACCDataXYZ[0],pACCDataXYZ[1],pACCDataXYZ[2],
                                                                    pMAGDataXYZ[0],pMAGDataXYZ[1],pMAGDataXYZ[2],
                                                                    pGyroDataXYZ[0],pGyroDataXYZ[1],pGyroDataXYZ[2]);
                        nsapi_size_t bytes_to_send = strlen(data);
                        nsapi_size_or_error_t bytes_sent = 0;
                        printf("\r\nSending sensor message: \r\n%s", data);
                        while (bytes_to_send) {
                            bytes_sent = _socket.send(data + bytes_sent, bytes_to_send);
                            printf("sent %d bytes\r\n", bytes_sent);
                            bytes_to_send -= bytes_sent;
                        }
                        ThisThread::sleep_for(1000);
                    }
                    if(button_switch != 2)
                        break;
                }
                else if (button_switch == 3) {
                    while(button_switch == 3);
                    if (button_switch != 3)
                        break;
                }
            }
            led_sem.release();
        }
    }


private:


    bool resolve_hostname(SocketAddress &address)
    {
        const char hostname[] = MBED_CONF_APP_HOSTNAME;

        /* get the host address */
        printf("\nResolve hostname %s\r\n", hostname);
        nsapi_size_or_error_t result = _net->gethostbyname(hostname, &address);
        if (result != 0) {
            printf("Error! gethostbyname(%s) returned: %d\r\n", hostname, result);
            return false;
        }

        printf("%s address is %s\r\n", hostname, (address.get_ip_address() ? address.get_ip_address() : "None") );

        return true;
    }

    bool send_http_request()
    {
        /* loop until whole request sent */
        const char buffer[] = "GET / HTTP/1.1\r\n"
                              "Host: ifconfig.io\r\n"
                              "Connection: close\r\n"
                              "\r\n";

        nsapi_size_t bytes_to_send = strlen(buffer);
        nsapi_size_or_error_t bytes_sent = 0;

        printf("\r\nSending message: \r\n%s", buffer);

        while (bytes_to_send) {
            bytes_sent = _socket.send(buffer + bytes_sent, bytes_to_send);
            if (bytes_sent < 0) {
                printf("Error! _socket.send() returned: %d\r\n", bytes_sent);
                return false;
            } else {
                printf("sent %d bytes\r\n", bytes_sent);
            }

            bytes_to_send -= bytes_sent;
        }
        printf("Complete message sent\r\n");
        return true;
    }

    bool receive_http_response()
    {
        char buffer[MAX_MESSAGE_RECEIVED_LENGTH];
        int remaining_bytes = MAX_MESSAGE_RECEIVED_LENGTH;
        int received_bytes = 0;

        /* loop until there is nothing received or we've ran out of buffer space */
        nsapi_size_or_error_t result = remaining_bytes;
        while (result > 0 && remaining_bytes > 0) {
            result = _socket.recv(buffer + received_bytes, remaining_bytes);
            if (result < 0) {
                printf("Error! _socket.recv() returned: %d\r\n", result);
                return false;
            }

            received_bytes += result;
            remaining_bytes -= result;
        }

        /* the message is likely larger but we only want the HTTP response code */

        printf("received %d bytes:\r\n%.*s\r\n\r\n", received_bytes, strstr(buffer, "\n") - buffer, buffer);

        return true;
    }

    void wifi_scan()
    {
        WiFiInterface *wifi = _net->wifiInterface();

        WiFiAccessPoint ap[MAX_NUMBER_OF_ACCESS_POINTS];

        /* scan call returns number of access points found */
        int result = wifi->scan(ap, MAX_NUMBER_OF_ACCESS_POINTS);

        if (result <= 0) {
            printf("WiFiInterface::scan() failed with return value: %d\r\n", result);
            return;
        }

        printf("%d networks available:\r\n", result);

        for (int i = 0; i < result; i++) {
            printf("Network: %s secured: %s BSSID: %hhX:%hhX:%hhX:%hhx:%hhx:%hhx RSSI: %hhd Ch: %hhd\r\n",
                   ap[i].get_ssid(), get_security_string(ap[i].get_security()),
                   ap[i].get_bssid()[0], ap[i].get_bssid()[1], ap[i].get_bssid()[2],
                   ap[i].get_bssid()[3], ap[i].get_bssid()[4], ap[i].get_bssid()[5],
                   ap[i].get_rssi(), ap[i].get_channel());
        }
        printf("\r\n");
    }

    void print_network_info()
    {
        /* print the network info */
        SocketAddress a;
        _net->get_ip_address(&a);
        printf("IP address: %s\r\n", a.get_ip_address() ? a.get_ip_address() : "None");
        _net->get_netmask(&a);
        printf("Netmask: %s\r\n", a.get_ip_address() ? a.get_ip_address() : "None");
        _net->get_gateway(&a);
        printf("Gateway: %s\r\n", a.get_ip_address() ? a.get_ip_address() : "None");
    }

private:
    NetworkInterface *_net;

#if MBED_CONF_APP_USE_TLS_SOCKET
    TLSSocket _socket;
#else
    TCPSocket _socket;
#endif // MBED_CONF_APP_USE_TLS_SOCKET
};

void button_pressed()
{
    if (button_switch == -1) {
        ++button_switch;
        led_sem.release();
    }
}

void button_released()
{
    if(button_switch >= 3){
        button_switch = 0;
    }
    ++button_switch;
}


int main() {
    printf("\r\nStarting socket demo\r\n\r\n");

#ifdef MBED_CONF_MBED_TRACE_ENABLE
    mbed_trace_init();
#endif

    SocketDemo *example = new SocketDemo();
    MBED_ASSERT(example);
    example->run();

    //example->test();
    const int a1 = 1;
    const int a2 = 2;
    const int a3 = 3;

    button.fall(&button_pressed); 
    button.rise(&button_released);
    t1.start(callback([=] { example->thread(); }));
    t2.start(callback([=] { example->thread(); }));
    t3.start(callback([=] { example->thread(); }));

    while (1);
    
}