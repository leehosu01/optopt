#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:37:10 2021

@author: map
"""
import asyncio
class agent:
    def __init__(self, manager):
        self.manager = manager
        pass

    async def _start(self):
        asyncio.ensure_future(async_foo())
    def start(self):
        asyncio.run(self._start())