MODSOM Generator — User Manual
1. Overview

modsom_generator is a test and simulation tool that generates synthetic MOD-SOM data compatible with the MOD-SOM parser and plotting application.

It produces data using the exact same binary record structure as the real system, making it suitable for:

Parser validation

Plotting and GUI development

Timing and latency tests

Spectral analysis verification

Each instrument channel is generated as a 1 Hz sinusoidal signal with unit amplitude.

2. Supported Output Modes

The generator can write data using one common backend to:

Binary file (.modraw)

Serial port (live stream)

TCP stream (network stream)

All modes generate identical data, differing only in how bytes are transported.

3. Supported Instruments

The following instrument tags are currently generated:

Instrument	Tag	Status
Epsilometer	EFE4	Fully implemented
Travel-Time Velocimetry	TTV1, TTV2, TTV3	Fully implemented
VectorNav VN-100	VNAV	$VNMAR ASCII samples
SeaBird CTD	SB49, SB41	Placeholder
ECO-Puck	ECOP	Placeholder
SOM controller	SOM3	Placeholder
4. Data Characteristics
4.1 Time

Record timestamp:
POSIX time in milliseconds, encoded as a 16-byte hexadecimal number

Sample timestamps:
Instrument-specific, always consistent with record time

4.2 Signals

All channels are:

Sinusoidal

1 Hz frequency

Amplitude = 1 (instrument units)

Different channels have phase offsets for visual clarity

5. Command-Line Usage

Run the generator with:

python modsom_generator.py [OPTIONS]

6. File Output Mode

Generate a .modraw file for offline replay.

python modsom_generator.py --file modsom_0.modraw --duration 120

Arguments

--file PATH
Output file name

--duration SECONDS
Length of generated data (default: 60 s)

Behavior

Generator runs for the specified duration

Writes a finite MOD-SOM binary file

File can be replayed by the parser

Replay
python parser_plotter.py --file modsom_0.modraw

7. Serial Streaming Mode

Stream data over a serial port (hardware-like behavior).

python modsom_generator.py --serial /dev/tty.usbserial-XXXX --baud 115200

Arguments

--serial PORT
Serial device

--baud RATE
Baudrate (default: 115200)

Behavior

Runs continuously until interrupted

Mimics embedded acquisition hardware

Sends som.stop\r\n on clean shutdown

Parser
python parser_plotter.py --serial /dev/tty.usbserial-XXXX --baud 115200

8. TCP Streaming Mode

Stream MOD-SOM data over a TCP socket.

python modsom_generator.py --tcp 0.0.0.0:9000

Arguments

--tcp HOST:PORT
IP address and port to bind the TCP server

Behavior

Generator acts as a TCP server

One client connects at a time

Data is streamed continuously

Stream stops when client disconnects or generator exits

Parser
python modsom_generator.py --tcp-listen 0.0.0.0:9000 --duration 9999 --realtime

⚠️ TCP is point-to-point.
Only one parser can connect at a time.

9. Stopping the Generator

The generator stops when:

Ctrl-C is pressed

The serial or TCP connection closes

The duration expires (file mode)

On shutdown:

Serial output receives som.stop\r\n

TCP socket is closed cleanly

Files are flushed and closed

10. Typical Use Cases

✅ Validate new parsing logic

✅ Test real-time PyQtGraph plotting

✅ Debug header / payload checksums

✅ Verify time synchronization

✅ Demonstrate the full pipeline without hardware

11. Limitations

Data is synthetic (no real noise or turbulence)

TCP mode supports one client only

Some instruments are placeholders

UDP / multicast is not implemented yet

12. Summary

modsom_generator is a protocol-faithful MOD-SOM simulator that allows you to test your entire acquisition and visualization chain — file, serial, or TCP — without deploying hardware.

If the parser works with this generator, it will work with the real system.
