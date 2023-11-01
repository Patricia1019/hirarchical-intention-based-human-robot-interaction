#! /usr/bin/env python3

###
# KINOVA (R) KORTEX (TM)
#
# Copyright (c) 2021 Kinova inc. All rights reserved.
#
# This software may be modified and distributed
# under the terms of the BSD 3-Clause license.
#
# Refer to the LICENSE file for details.
#
###

import sys
import os
import time
import threading

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient


from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

import utilities
from modular_control import ModuleController
import pdb

def receiver(conn):
    # Parse arguments
    # args = utilities.parseConnectionArguments()
    while 1:
        msg = conn.recv()
        # msg = conn
        print(msg)
        if msg == "get long tubes":
        # if msg.value == 1:
            args = utilities.Args()
            
            # Create connection to the device and get the router
            with utilities.DeviceConnection.createTcpConnection(args) as router:

                # Create required services
                base = BaseClient(router)
                base_cyclic = BaseCyclicClient(router)
                controller = ModuleController(router,base,base_cyclic)
                # Example core
                success = True
                success = controller.making_module_T(success)
                # return 0 if success else 1
        elif msg == "break":
        # elif msg.value == 0:
            break

if __name__ == "__main__":
    # exit(main())
    main()
