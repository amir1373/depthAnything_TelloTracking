import socket

# Set up the UDP receiver
udp_ip = "127.0.0.1"  # Localhost (can change to any IP)
udp_port = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((udp_ip, udp_port))

print(f"Listening for bounding box data on {udp_ip}:{udp_port}")

while True:
    # Receive the message (which should contain the bounding box data)
    message, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
    bbox_data = message.decode('utf-8')
    
    # Split the data into components (center_x, center_y, width, height)
    try:
        center_x, center_y, width, height = map(int, bbox_data.split(','))
        print(f"Received Bounding Box: Center=({center_x}, {center_y}), Width={width}, Height={height}")
    except ValueError:
        print("Invalid data received.")
