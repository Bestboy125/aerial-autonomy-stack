from pymavlink import mavutil
import argparse

def main():
    parser = argparse.ArgumentParser(description='Request MAVLink data streams from autopilot')
    parser.add_argument('--device', type=str, default='/dev/ttyTHS1',
                        help='Serial device path (default: /dev/ttyTHS1)')
    parser.add_argument('--baudrate', type=int, default=921600,
                        help='Baudrate (default: 921600)')
    parser.add_argument('--target-system', type=int, default=1,
                        help='Target system ID (default: 1)')
    parser.add_argument('--target-component', type=int, default=1,
                        help='Target component ID (default: 1)')
    parser.add_argument('--rate', type=int, default=4,
                        help='Stream rate in Hz (default: 4)')
    args = parser.parse_args()

    master = mavutil.mavlink_connection(args.device, baud=args.baudrate)
    master.wait_heartbeat()
    print(f"Heartbeat received from target_system {master.target_system}")

    print(f"Requesting data streams at {args.rate} Hz from target_system {args.target_system}, target_component {args.target_component}")
    master.mav.request_data_stream_send(
        args.target_system, 
        args.target_component, 
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 
        args.rate, 
        1 # 1 = start, 0 = stop
    )
    print("Data streams requested")

if __name__ == "__main__":
    main()
