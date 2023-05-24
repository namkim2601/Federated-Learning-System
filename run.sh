if [ "$1" = '0' ]; then
    python3 500436282_COMP3221_FLServer.py 6000 0 &
elif [ "$1" = '1' ]; then
    python3 500436282_COMP3221_FLServer.py 6000 1 &
else
    echo "The first arg must be 0 or 1 - It determines the FLServer sub_client mode"
    exit 
fi
sleep 1
if [ "$2" = '0' ]; then
    python3 500436282_COMP3221_FLClient.py client1 6001 0 &
    python3 500436282_COMP3221_FLClient.py client2 6002 0 &
    python3 500436282_COMP3221_FLClient.py client3 6003 0 &
    python3 500436282_COMP3221_FLClient.py client4 6004 0 &
    python3 500436282_COMP3221_FLClient.py client5 6005 0 &
elif [ "$2" = '1' ]; then
    python3 500436282_COMP3221_FLClient.py client1 6001 1 &
    python3 500436282_COMP3221_FLClient.py client2 6002 1 &
    python3 500436282_COMP3221_FLClient.py client3 6003 1 &
    python3 500436282_COMP3221_FLClient.py client4 6004 1 &
    python3 500436282_COMP3221_FLClient.py client5 6005 1 &
else
    echo "The second arg must be 0 or 1 - It determines the FLClient opt_method"
    ps aux | grep python | grep -v "grep python" | awk '{print $2}' | xargs kill -9 # Kill server
    exit 
fi