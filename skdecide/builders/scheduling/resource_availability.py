from __future__ import annotations
from typing import List, Union, Dict, Optional
from skdecide.builders.scheduling.scheduling_domains_modelling import SchedulingEvent
from enum import Enum

__all__ = ['UncertainResourceAvailabilityChanges', 'DeterministicResourceAvailabilityChanges',
           'WithoutResourceAvailabilityChange']


class UncertainResourceAvailabilityChanges:
    """A domain must inherit this class if the availability of its resource vary in an uncertain way over time.
    This class enables the definition of resource change events."""

    def _get_original_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        """Return the resource availability (int) for the given resource
        (either resource type or resource unit) at the given time."""
        raise NotImplementedError

    def get_original_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        """Return the resource availability (int) for the given resource
        (either resource type or resource unit) at the given time."""
        return self._get_original_quantity_resource(resource, time, **kwargs)

    def get_all_planned_resource_changes(self, resource: str) -> List[int]:
        """Return the time steps of the planned resource changes as a list.
         A timestep t is in the list if the resource quantity at t differs from the resource quantity at t-1
         in the original quantity."""
        vals = []
        for t in range(1, self._get_max_horizon()):
            if self.get_original_quantity_resource(resource, t) != self.get_original_quantity_resource(resource, t-1):
                vals.append(t)
        return vals


    def _get_next_resource_change_time_distribution(self, resource: str, currenttime: int, previous_resource_event: Optional[(int, int)] = None):
        """ Return a Distribution defining the probability to experience a resource availability event at any time step.
        If a time step is not in the distribution, assume its probability to experience an event is 0.
        The current time step and the time step of the previous resource event given as input can be used to define this distribution.
        E.g. A distribution for 4 time step (from 13 to 16) can be defined as:
            Distribution([(13, 0.4), (14, 0.3), (15, 0.2), (16, 0.1)]).
        """
        raise NotImplementedError

    def get_next_resource_change_time_distribution(self, resource: str, currenttime: int, previous_resource_event: Optional[SchedulingEvent] = None):
        """ Return a Distribution defining the probability to experience a resource availability event at any time step.
        If a time step is not in the distribution, assume its probability to experience an event is 0.
        The current time step and the time step of the previous resource event given as input can be used to define this distribution.
        E.g. A distribution for 4 time step (from 13 to 16) can be defined as:
            Distribution([(13, 0.4), (14, 0.3), (15, 0.2), (16, 0.1)]).
        """
        return self._get_next_resource_change_time_distribution(resource, currenttime, previous_resource_event)

    def _get_next_resource_change_delta_distribution(self, resource: str, change_time: int, previous_resource_event: Optional[SchedulingEvent] = None):
        """ Return a Distribution defining the magnitude of a resource availability change.
        The time step of the event and the information about the previous event (time and magnitude) can be used to define this distribution.
        E.g. A distribution for 4 levels of changes can be defined as:
            Distribution([(-2, 0.1), (-1, 0.4), (1, 0.4), (2, 0.1)]).
        """
        raise NotImplementedError

    def get_next_resource_change_delta_distribution(self, resource: str, change_time: int,
                                                     previous_resource_event: Optional[(int, int)] = None):
        """ Return a Distribution defining the magnitude of a resource availability change.
        The time step of the event and the information about the previous event (time and magnitude) can be used to define this distribution.
        E.g. A distribution for 4 levels of changes can be defined as:
            Distribution([(-2, 0.1), (-1, 0.4), (1, 0.4), (2, 0.1)]).
        """
        return self._get_next_resource_change_delta_distribution(resource, change_time, previous_resource_event)

    def _get_quantity_resource(self, resource: str, time: int, previous_resource_event: Optional[(int, int)] = None, next_resource_event: Optional[(int, int)] = None):
        """ Return the quantity of resource of a given type available at a given time,
        potentially considering the previous and next resource change events.
        """
        raise NotImplementedError

    def get_quantity_resource(self, resource: str, time: int, previous_resource_event: Optional[(int, int)] = None, next_resource_event: Optional[(int, int)] = None):
        """ Return the quantity of resource of a given type available at a given time,
        potentially considering the previous and next resource change events.
        """
        return self._get_quantity_resource(resource, time, previous_resource_event, next_resource_event)


class DeterministicResourceAvailabilityChanges(UncertainResourceAvailabilityChanges):
    """A domain must inherit this class if the availability of its resource vary in a deterministic way over time."""

    def _get_original_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        """Return the resource availability (int) for the given resource
        (either resource type or resource unit) at the given time."""
        raise NotImplementedError

    def _get_next_resource_change_time_distribution(self, resource: str, currenttime: int, previousresourcehangetime: Optional[int]):
        """E.g. Distribution(int, float) """
        return None

    def _get_next_resource_change_delta_distribution(self, resource: str, change_time: int, previous_resource_event: Optional[(int, int)] = None):
        """"""
        return 0

    def _get_quantity_resource(self, resource: str, time: int, previous_resource_event: Optional[(int, int)] = None, next_resource_event: Optional[(int, int)] = None, **kwargs) -> int:
        """Return the resource availability (int) for the given resource
         (either resource type or resource unit) at the given time."""
        return self._get_original_quantity_resource(resource=resource, time=time, **kwargs)

    # def _sample_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
    #     """Sample an amount of resource availability (int) for the given resource
    #      (either resource type or resource unit) at the given time. This number should be the sum of the number of
    #      resource available at time t and the number of resource of this type consumed so far)."""
    #     return self.get_quantity_resource(resource, time, **kwargs)


class WithoutResourceAvailabilityChange(DeterministicResourceAvailabilityChanges):
    """A domain must inherit this class if the availability of its resource does not vary over time."""

    def _get_fixed_quantity_resource(self, resource: str, **kwargs) -> int:
        """Return the resource availability (int) for the given resource (either resource type or resource unit)."""
        raise NotImplementedError

    def get_fixed_quantity_resource(self, resource: str, **kwargs) -> int:
        """Return the resource availability (int) for the given resource (either resource type or resource unit)."""
        return self._get_fixed_quantity_resource(resource=resource, **kwargs)

    def _get_original_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
        """Return the resource availability (int) for the given resource
        (either resource type or resource unit) at the given time."""
        return self._get_fixed_quantity_resource(resource)

    def get_all_planned_resource_changes(self, resource: str) -> List[int]:
        """Return the time steps of the planned resource changes as a list.
         A timestep t is in the list if the resource quantity at t differs from the resource quantity at t-1
         in the original quantity."""
        return []

    # def _get_quantity_resource(self, resource: str, time: int, previous_resource_event: Optional[(int, int)] = None, next_resource_event: Optional[(int, int)] = None, **kwargs) -> int:
    #     """Sample an amount of resource availability (int) for the given resource
    #      (either resource type or resource unit) at the given time. This number should be the sum of the number of
    #      resource available at time t and the number of resource of this type consumed so far)."""
    #     return self._get_fixed_quantity_resource(resource)

    # def _sample_quantity_resource(self, resource: str, time: int, **kwargs) -> int:
    #     """Sample an amount of resource availability (int) for the given resource
    #      (either resource type or resource unit) at the given time. This number should be the sum of the number of
    #      resource available at time t and the number of resource of this type consumed so far)."""
    #     return self._get_fixed_quantity_resource(resource)
