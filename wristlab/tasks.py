"""Task selection delegates every physical transfer to actions.relocate."""
import numpy as np
import time
from .types import ActionResult, Placement


MODES = ('relocate','stow','extract','collect_to_box','empty_box','sort_colors')


class SortingBlocked(ValueError):
    def __init__(self, reasons):
        self.reasons = reasons
        super().__init__('sorting has no accessible destination: '+ '; '.join(
            f'{object_id}: {reason}' for object_id,reason in reasons.items()))


def box_for_color(color, boxes):
    if color not in ('green','yellow','red','blue'):
        raise ValueError('sorting requires a confidently classified supported cube color')
    ordered = sorted(boxes.values(),key=lambda b:np.mean(np.array(b.evidence['polygon']),axis=0)[0])
    if len(ordered) != 2:
        raise ValueError('sorting requires exactly two independently measured boxes')
    return ordered[0].id if color in ('green','yellow') else ordered[1].id


def placement_in_box(scene, object_id, box_id):
    obj = scene.state.objects[object_id]
    patches = scene.find_free_box_slots(box_id,obj.dimensions)
    if not patches:
        raise ValueError('no free slot with finger clearance remains in box')
    patch = patches[0]
    pose = obj.pose.copy()
    pose[:3,3] = [*np.mean(patch.polygon_xy_m,axis=0),patch.height_m+obj.dimensions[2]/2]
    return Placement(pose,box_id)


def next_transfer(mode, scene, selected=None, destination=None, excluded=()):
    objects = [o for o in scene.state.objects.values() if o.id not in excluded]
    if mode == 'relocate':
        if selected is None or destination is None:
            raise ValueError('select an observed object and a destination')
        return selected,placement_in_box(scene,selected,destination)
    if mode in ('stow','collect_to_box','pack_box'):
        if destination not in scene.state.boxes:
            raise ValueError('select an observed box')
        outside = [o for o in objects if not any(o.id in b.occupied_slots for b in scene.state.boxes.values())]
        if not outside:
            return None
        obj = min(outside,key=lambda o:np.linalg.norm(o.pose[:2,3]-scene.state.boxes[destination].pose[:2,3]))
        return obj.id,placement_in_box(scene,obj.id,destination)
    if mode in ('extract','empty_box'):
        box = scene.state.boxes[destination]
        if not box.occupied_slots:
            return None
        graph = scene.build_support_graph()
        available = [scene.state.objects[key] for key in box.occupied_slots
                     if key not in excluded and not graph[key] and scene.state.objects[key].status!='held']
        if not available:
            if any(key not in excluded for key in box.occupied_slots):
                raise ValueError('box contains objects with no top-accessible extraction; clear the support graph first')
            return None
        obj = next((obj for obj in available if obj.id==selected),max(available,key=lambda obj:obj.pose[2,3]))
        slots = scene.find_free_table_slots(obj.dimensions)
        if not slots:
            raise ValueError('no independently surveyed free table support is available')
        support = slots[0]
        pose = obj.pose.copy()
        pose[:3,3] = [*np.mean(support['patch'].polygon_xy_m,axis=0),support['height_m']+obj.dimensions[2]/2]
        return obj.id,Placement(pose,support['id'])
    if mode == 'sort_colors':
        if not scene.state.objects:
            raise ValueError('sorting cannot confirm completion without observed cubes')
        wrong = [o for o in objects if o.id not in scene.state.boxes[box_for_color(o.color,scene.state.boxes)].occupied_slots]
        if not wrong:
            if any(not o.evidence.get('metric_qualified') or time.time()-o.timestamp>.5 for o in objects):
                raise ValueError('sorting completion requires fresh measured cube positions')
            return None
        graph = scene.build_support_graph()
        available = [obj for obj in wrong if not graph[obj.id] and obj.status!='held']
        failures = {obj.id:'support graph blocks access' for obj in wrong if graph[obj.id] or obj.status=='held'}
        for obj in sorted(available,key=lambda obj:-obj.pose[2,3]):
            try:
                return obj.id,placement_in_box(scene,obj.id,box_for_color(obj.color,scene.state.boxes))
            except ValueError as exc:
                failures[obj.id] = str(exc)
        raise SortingBlocked(failures)
    raise ValueError('experiment has not reached implementation/qualification stage')


def run_task(mode, scene, actions, refresh, check_cancel, selected=None, destination=None, *, dry=False):
    options = {'dry':True} if dry else {}
    if mode == 'relocate':
        check_cancel()
        refresh()
        return [actions.relocate(*next_transfer(mode,scene,selected,destination),**options)]
    results, deferred = [], set()
    initial_count = len(scene.state.objects)
    for _ in range(max(1,initial_count)):
        check_cancel()
        refresh()
        try:
            item = next_transfer(mode,scene,selected,destination,deferred)
        except SortingBlocked as exc:
            return results+[ActionResult(False,reason,object_id,scene.state,
                {'motion_started':False,'physical_verified':False,'failure_kind':'PLANNING_OR_OBSERVATION'})
                for object_id,reason in exc.reasons.items()]
        if item is None:
            return results
        object_id,placement = item
        if object_id in deferred:
            return results
        result = actions.relocate(object_id,placement,**options)
        results.append(result)
        # A dry check does not change occupancy. Do not keep checking the
        # first successful cube while silently skipping the other colors.
        if dry or not result.success:
            deferred.add(object_id)
            if result.evidence.get('critical') or (result.evidence.get('motion_started') and not result.evidence.get('recovered')):
                return results
    return results
