# LPU & Punjab Regional Transit Domain Model

Domain model for forecasting passenger demand and assisting fleet capacity decisions 24 hours prior to scheduled departure along the Phagwara – Jalandhar – Lovely Professional University (LPU) NH-44 corridor.

## Language

**Trip**:
A single scheduled bus departure on a designated Punjab transit corridor at a concrete date and time.
_Avoid_: Ride, journey, run

**Route**:
A predefined physical transit corridor connecting specific university gates, railway junctions, and city bus stands.
_Avoid_: Line, service, pathway

**Passenger Demand**:
The total number of students, faculty, and regional commuters seeking to board a specific Trip, independent of bus capacity limits.
_Avoid_: Headcount, ridership, ticket sales

**Capacity**:
The maximum safe seating and standing threshold available on the vehicle assigned to a Trip.
_Avoid_: Bus size, vehicle limit, quota

**Occupancy Rate**:
The ratio of Passenger Demand to assigned vehicle Capacity for a Trip, expressed as a percentage or fraction.
_Avoid_: Load factor, utilization percentage

**Action Flag**:
A classification assigned to a Trip indicating whether fleet reallocation, bus downsizing, or an extra relief shuttle is required.
_Avoid_: Status, alert level, dispatch tag

**Forecast Horizon**:
The elapsed time between when a prediction is generated and the Trip departure time (fixed at 24 hours).
_Avoid_: Lead time, lookahead window

**Transport Coordinator**:
The human-in-the-loop operational decision-maker at the university and regional depot responsible for allocating buses and approving relief runs.
_Avoid_: Bus manager, fleet admin, controller

## Corridors

**`LPU_JAL_101` (LPU Main Gate → Jalandhar City Bus Stand)**:
The primary student and faculty commuter corridor running along NH-44 via Rama Mandi with high morning and evening rush volume.

**`PHG_LPU_202` (Phagwara Junction Railway Station → LPU Campus)**:
Express railway feeder service connecting long-distance train arrivals to campus hostels, spiking heavily on Sunday evenings and Monday mornings.

**`LPU_MDL_303` (LPU Campus → Model Town, Jalandhar)**:
Evening and weekend shopping and dining transit corridor connecting campus residents to urban recreation centers.

**`INT_CAMPUS_404` (LPU Main Gate → Block 34 / Uni-Hospital)**:
High-frequency internal university perimeter shuttle with severe demand bursts synchronized with academic timetable class dismissals.

**`JAL_ASR_505` (Jalandhar Cantt → Phagwara GT Road Feeder)**:
Suburban residential feeder line providing steady, lower-density connections across the Grand Trunk Road.
