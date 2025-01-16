from pydantic import BaseModel

class Account(BaseModel):
    name: 
    facilities: 
    groups: 

class Facility(BaseModel):
    account: 
    name: 
    groups: 
    machines: 

class MachineGroup(BaseModel):
    account: 
    name: 
    description: 
    machines: 

class Machine(BaseModel):
    name: 
    facility: 
    group: 
    mtype: 
    spec: 
    monitors: 

class Monitor(BaseModel):
    machine: 
    name: 
    description: 
    device: 

class MonitoringDevice(BaseModel):
    monitor: 
    mac_id: 
    hardware: 

