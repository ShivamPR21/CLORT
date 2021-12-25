'''
Copyright (C) 2021  Shiavm Pandey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from queue import Queue
from typing import Any, bool


class ModifiedQueue(Queue):

    def __init__(self, maxsize: int = 10, item = None) -> None:
        super().__init__(maxsize=maxsize)
        if (item) : self.put(item)

    def put(self, item) -> bool:
        if (super().full()) :
            super().get()
        super().put(item)

    def get(self) :
        if (super().empty()):
            return None
        super().get()

    def __getitem__(self, key : int) -> Any:
        if (super().qsize() > abs(key)):
            return super()[key]

        return None
