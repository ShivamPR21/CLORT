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

from typing import Any


class StateEvolution:

    def __init__(self,
                 state_dim : int = 3,
                 obs_dim : int = 3) -> None:
        self.state_dim, self.obs_dim = state_dim, obs_dim

    def predict(self) -> Any:
        pass

    def update(self) -> Any:
        pass

    def state_covar(self) -> Any:
        pass

    def obs_covar(self) -> Any:
        pass

    def process_noise(self) -> Any:
        pass
