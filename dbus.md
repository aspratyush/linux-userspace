# DBus
v0.1

NOMENCLATURE
--------------------

#### Service

- program offering IPC over a common bus
- service is identified by its "reverse domain name" over the bus

#### Client / Peer

- program that makes use of some IPC provided by a service on the bus

#### Object Path

- an identifier for a type of service (similar to class objects)
- passing around memory addresses of service APIs doesnt make sense
- Hence, a file-system like path is provided
- e.g. - /org/freedesktop/logind

#### Interfaces

- collection of signals, properties, methods exposed by an "object path"
- this is something similar to class members (variables and functions)
- identified by reverse domain name (no restriction on name being similar to service)
- some interfaces are standardized, and are available on all bus-connected services
- e.g - org.freedesktop.DBus.Introspectable, org.freedesktop.DBus.Peer

#### THUS,
1. A utility (sample) exposes a METHOD (GetData) at an INTERFACE (com.example.psahay.sample.data), available at an OBJECT PATH (com/example/psahay/sample) of a service (com.example.psahay.sample)
2. an object path may have a collection of interfaces, methods, signals and properties

INTERFACE MEMBERS
---------------------

#### Method

- Sync calls provided by a service
- Signature : params taken by a method

#### Signal

- one-to-one / many-to-one broadcast
- async notification of peers

#### Property

- variables that can be read by a client
 
BUS MONITORING (busctl)
------------------------

#### Naming convention

- Low-level : Unique names (similar to IP)
- High-level : well-known service names (similar to DNS host names)

#### busctl tree <serviceName>

- objects available in a service

#### busctl introspect <serviceName> <objectPath>

#### bustcl call <service> <object> method


#### To execute the sample files, execute the following:   
$ python main.py
