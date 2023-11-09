
#include "mbed.h"
#include "wifi_helper.h"
#include "mbed-trace/mbed_trace.h"
#include "stm32l475e_iot01_magneto.h"
#include "stm32l475e_iot01_gyro.h"
#include "stm32l475e_iot01_accelero.h"

#if MBED_CONF_APP_USE_TLS_SOCKET
#include "root_ca_cert.h"

#ifndef DEVICE_TRNG
#error "mbed-os-example-tls-socket requires a device which supports TRNG"
#endif
#endif // MBED_CONF_APP_USE_TLS_SOCKET
volatile int button_switch = 0;

class SocketDemo {
    static constexpr size_t MAX_NUMBER_OF_ACCESS_POINTS = 10;
    static constexpr size_t MAX_MESSAGE_RECEIVED_LENGTH = 100;

#if MBED_CONF_APP_USE_TLS_SOCKET
    static constexpr size_t REMOTE_PORT = 443; // tls port
#else
    static constexpr size_t REMOTE_PORT = 80; // standard HTTP port
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

            /* in this example we use credentials configured at compile time which are used by
             * NetworkInterface::connect() but it's possible to do this at runtime by using the
             * WiFiInterface::connect() which takes these parameters as arguments */
        }

        /* connect will perform the action appropriate to the interface type to connect to the network */

        printf("Connecting to the network...\r\n");

        nsapi_size_or_error_t result = _net->connect();
        if (result != 0) {
            printf("Error! _net->connect() returned: %d\r\n", result);
            return;
        }

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

        Semaphore press_sem(0);
        Thread p1;
        Thread p2;
        InterruptIn button(BUTTON1);
        void button_pressed(){
            ++button_switch;
            if (button_switch == 1){
                press_sem.release();
            }
        }
        void button_released(){
            ++button_switch;
        }
        const int a1 = 1;
        const int a2 = 2;
        button.fall(&button_pressed);
        p1.start(callback(socket_thread, (void *)&a1));
        button.rise(&button_released);
        p1.start(callback(socket_thread, (void *)&a2));
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

    void sensor_socket() {

        float sensor_value = 0;
        int16_t pACCDataXYZ[3] = {0};
        int16_t pmagDataXYZ[3] = {0};
        float pGyroDataXYZ[3] = {0};
        
        BSP_MAGNETO_Init();
        BSP_GYRO_Init();
        BSP_ACCELERO_Init();

        while(1){
            BSP_ACCELERO_AccGetXYZ(pACCDataXYZ);
            BSP_MAGNETO_GetXYZ(pmagDataXYZ);
            BSP_GYRO_GetXYZ(pGyroDataXYZ);

            char data[700];
            sprintf(data,"%d %d %d %d %d %d %f %f %f",pACCDataXYZ[0],pACCDataXYZ[1],pACCDataXYZ[2],
                                                    pmagDataXYZ[0],pmagDataXYZ[1],pmagDataXYZ[2],
                                                    pGyroDataXYZ[0],pGyroDataXYZ[1],pGyroDataXYZ[2]);
            nsapi_size_t bytes_to_send = strlen(data);
            nsapi_size_or_error_t bytes_sent = 0;

            while (bytes_to_send) {
                bytes_sent = _socket.send(data + bytes_sent, bytes_to_send);
                if (bytes_sent < 0)
                    printf("Error! _socket.send() returned: %d\r\n", bytes_sent);
                else 
                    printf("sent %d bytes\r\n", bytes_sent);
                bytes_to_send -= bytes_sent;
            }
            ThisThread::sleep_for(1000);
        }
    }

    void socket_thread(void const *name) {
        while (1) {
            press_sem.acquire();
            while (1) {
                if (*((int*)name) == 1) { 
                    sensor_socket();
                    if(button_switch % 2 == 0)
                        break;
                }
                else if (*((int*)name) == 2) {
                    //another_model();
                    if (button_switch % 2 == 1)
                        break;
                }
            }
            press_sem.release();
        }
    }


    

private:
    NetworkInterface *_net;

#if MBED_CONF_APP_USE_TLS_SOCKET
    TLSSocket _socket;
#else
    TCPSocket _socket;
#endif // MBED_CONF_APP_USE_TLS_SOCKET
};



int main() {
    printf("\r\nStarting socket demo\r\n\r\n");

#ifdef MBED_CONF_MBED_TRACE_ENABLE
    mbed_trace_init();
#endif

    SocketDemo *example = new SocketDemo();
    MBED_ASSERT(example);

    example->run();

    return 0;
}
//用socket傳加速度、磁力計、陀螺儀、模式。