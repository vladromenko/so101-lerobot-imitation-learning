"""Task selection delegates every physical transfer to actions.relocate."""
import numpy as np
from .types import Placement


def box_for_color(color, boxes):
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
        available = [key for key in box.occupied_slots if key not in excluded]
        if not available:
            return None
        obj = scene.state.objects[available[-1]]
        slots = scene.find_free_table_slots(obj.dimensions)
        if not slots:
            raise ValueError('no independently surveyed free table support is available')
        support = slots[0]
        pose = obj.pose.copy()
        pose[:3,3] = [*np.mean(support['patch'].polygon_xy_m,axis=0),support['height_m']+obj.dimensions[2]/2]
        return obj.id,Placement(pose,support['id'])
    if mode == 'sort_colors':
        wrong = [o for o in objects if o.id not in scene.state.boxes[box_for_color(o.color,scene.state.boxes)].occupied_slots]
        if not wrong:
            return None
        obj = wrong[0]
        return obj.id,placement_in_box(scene,obj.id,box_for_color(obj.color,scene.state.boxes))
    raise ValueError('experiment has not reached implementation/qualification stage')


def run_task(mode, scene, actions, refresh, check_cancel, selected=None, destination=None):
    results, deferred = [], set()
    initial_count = len(scene.state.objects)
    for _ in range(initial_count):
        check_cancel()
        refresh()
        item = next_transfer(mode,scene,selected,destination,deferred)
        if item is None:
            return results
        object_id,placement = item
        if object_id in deferred:
            return results
        result = actions.relocate(object_id,placement)
        results.append(result)
        if not result.success:
            deferred.add(object_id)
            if result.evidence.get('motion_started') or result.evidence.get('critical'):
                return results
        if mode == 'relocate':
            return results
    return results
