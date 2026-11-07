"""Geometric task algorithms. Every transfer is performed by actions.relocate."""
from heapq import heappop, heappush
import numpy as np
from scipy.optimize import linear_sum_assignment
from .types import Placement


MODES = ('build_tower','unstack_tower','arrange_grid','build_color_pattern','pack_box','free_target','reset_scene','run_sequence')
DEFAULT_SEQUENCE = ('sort_colors','empty_box','build_tower','unstack_tower','reset_scene')


def on_table(scene, obj):
    return not any(obj.id in box.occupied_slots for box in scene.state.boxes.values())


def free_objects(scene, exclude=()):
    graph = scene.build_support_graph()
    return [o for o in scene.state.objects.values() if o.id not in exclude and not graph[o.id] and o.status != 'held']


def tower_chain(scene, base_id):
    graph = scene.build_support_graph()
    chain,seen = [base_id],{base_id}
    while graph[chain[-1]]:
        children = graph[chain[-1]]
        if len(children) != 1 or children[0] in seen:
            raise ValueError('ambiguous/branching support graph; observe another view')
        chain.append(children[0]);seen.add(children[0])
    return chain


def stack_step(scene, count=2, base_id=None, excluded=()):
    if count < 2:
        raise ValueError('tower needs at least two cubes')
    candidates = [o for o in scene.state.objects.values() if on_table(scene,o)]
    if not candidates:
        raise ValueError('no table-supported tower base')
    if base_id is None:
        center = np.mean([o.pose[:2,3] for o in candidates],axis=0)
        base_id = min(candidates,key=lambda o:np.linalg.norm(o.pose[:2,3]-center)).id
    chain = tower_chain(scene,base_id)
    if len(chain) >= count:
        return None
    available = free_objects(scene,(*chain,*excluded))
    if not available:
        raise ValueError('not enough accessible cubes for tower')
    support = scene.state.objects[chain[-1]]
    obj = min(available,key=lambda o:np.linalg.norm(o.pose[:2,3]-support.pose[:2,3]))
    target = support.pose.copy()
    target[2,3] = support.pose[2,3]+(support.dimensions[2]+obj.dimensions[2])/2
    return obj.id,Placement(target,'cube:'+support.id,min(.004,float(min(support.dimensions[:2]))/8),support.id)


def table_slot(scene, obj, excluded=()):
    slots = scene.find_free_table_slots(obj.dimensions)
    if not slots:
        raise ValueError('a current measured free table patch is required')
    available = [s for s in slots if s['id'] not in excluded]
    if not available:
        raise ValueError('no unreserved table patch remains')
    slot = min(available,key=lambda s:np.linalg.norm(np.mean(s['patch'].polygon_xy_m,axis=0)-obj.pose[:2,3]))
    target = obj.pose.copy()
    target[:3,3] = [*np.mean(slot['patch'].polygon_xy_m,axis=0),slot['height_m']+obj.dimensions[2]/2]
    return Placement(target,slot['id'])


def unstack_step(scene, destination=None, excluded=()):
    graph = scene.build_support_graph()
    supported = {child for children in graph.values() for child in children}
    top = [o for o in free_objects(scene,excluded) if o.id in supported]
    if not top:
        return None
    obj = max(top,key=lambda o:o.pose[2,3])
    if destination in scene.state.boxes:
        from .tasks import placement_in_box
        return obj.id,placement_in_box(scene,obj.id,destination)
    return obj.id,table_slot(scene,obj)


def grid_targets(scene, rows, columns, area=None):
    if rows < 1 or columns < 1 or rows*columns > len(scene.state.objects):
        raise ValueError('grid size exceeds observed cube count')
    objects = list(scene.state.objects.values())
    side = np.max([o.dimensions for o in objects],axis=0)
    # Finger space, derived from the measured cube footprint and native contact margin.
    from box_extract.stow import CUBE_KEEPOUT_LONG_M,CUBE_KEEPOUT_SHORT_M
    spacing = np.maximum(side[:2]*1.5,max(CUBE_KEEPOUT_LONG_M,CUBE_KEEPOUT_SHORT_M))
    if area is None:
        slots = scene.find_free_table_slots(side)
        if not slots:
            raise ValueError('grid area needs a measured free support surface')
        area = {'polygon_xy_m':slots[0]['patch'].polygon_xy_m,'height_m':slots[0]['height_m'],'surface_id':slots[0]['id']}
    polygon = np.asarray(area['polygon_xy_m'],float)
    lo,hi = polygon.min(0)+side[:2]/2,polygon.max(0)-side[:2]/2
    required = spacing*np.array([columns-1,rows-1])
    if np.any(hi-lo < required):
        raise ValueError('grid and finger clearances do not fit measured region')
    from scipy.spatial import ConvexHull
    hull = ConvexHull(polygon)
    origin = (lo+hi-required)/2
    targets = []
    for row in range(rows):
        for column in range(columns):
            xy = origin+spacing*np.array([column,row])
            corners = xy+np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*side[:2]/2
            if np.any(corners@hull.equations[:,:2].T+hull.equations[:,2] > 0):
                raise ValueError('grid corner leaves measured convex region')
            pose = np.eye(4);pose[:3,3] = [*xy,area['height_m']+side[2]/2]
            targets.append(Placement(pose,area['surface_id']))
    return targets


def assign_targets(scene, targets, colors=None):
    objects = free_objects(scene)
    if len(objects) < len(targets):
        raise ValueError('not enough top-accessible cubes for assignment')
    costs = np.array([[np.linalg.norm(o.pose[:3,3]-p.object_pose[:3,3]) for p in targets] for o in objects])
    if colors is not None:
        for i,obj in enumerate(objects):
            for j,color in enumerate(colors):
                if obj.color != color:
                    costs[i,j] = 1e6
    rows,columns = linear_sum_assignment(costs)
    if len(columns) != len(targets) or any(costs[i,j] >= 1e6 for i,j in zip(rows,columns)):
        raise ValueError('requested pattern contains unavailable colors')
    return [(objects[i].id,targets[j]) for i,j in sorted(zip(rows,columns),key=lambda pair:pair[1])]


def assignment_step(scene, assignments, excluded=()):
    pending = [(key,p) for key,p in assignments if key not in excluded and np.linalg.norm(scene.state.objects[key].pose[:3,3]-p.object_pose[:3,3]) > p.tolerance]
    if not pending:
        return None
    for key,placement in pending:
        obj = scene.state.objects[key]
        above = scene.build_support_graph()[key]
        obstacles = [other for other in scene.state.objects.values() if other.id != key and
            np.all(np.abs(other.pose[:2,3]-placement.object_pose[:2,3]) < (other.dimensions[:2]+obj.dimensions[:2])/2+.005)]
        if not obstacles and not above:
            return key,placement
    # An assignment cycle needs a measured temporary buffer; moving into an
    # occupied destination is never accepted as a shortcut.
    accessible = [key for key,_ in pending if not scene.build_support_graph()[key]]
    if not accessible:
        raise ValueError('assignment requires clearing the measured support graph first')
    key = accessible[0]
    return key,table_slot(scene,scene.state.objects[key])


def rearrangement_plan(scene, object_id):
    target = scene.state.objects[object_id]
    graph = scene.build_support_graph()
    above = graph[object_id]
    nearby = [o.id for o in scene.state.objects.values() if o.id != object_id and
        np.linalg.norm(o.pose[:2,3]-target.pose[:2,3]) < (max(o.dimensions[:2])+max(target.dimensions[:2]))/2+.008]
    blockers = tuple(dict.fromkeys([*above,*nearby]))
    if not blockers:
        return []
    if len(blockers) > 8:
        raise ValueError('rearrangement search exceeds bounded workspace capacity')
    slots = scene.find_free_table_slots(target.dimensions)
    if len(slots) < len(blockers):
        raise ValueError('not enough measured buffer positions for blockers')
    # A* states describe which blockers have moved. Support dependencies forbid
    # pulling a lower cube out from beneath an unmoved upper cube.
    full = (1<<len(blockers))-1
    frontier = [(len(blockers),0.,0,())]
    best = {0:0.}
    while frontier:
        _,cost,state,path = heappop(frontier)
        if state == full:
            return list(path)
        if cost <= best.get(state,float('inf')):
            for i,key in enumerate(blockers):
                supports_clear = all(child not in blockers or state&(1<<blockers.index(child)) for child in graph[key])
                if not state&(1<<i) and supports_clear:
                    obj = scene.state.objects[key]
                    slot_index = state.bit_count()
                    slot = slots[slot_index]
                    pose = obj.pose.copy();pose[:3,3] = [*np.mean(slot['patch'].polygon_xy_m,axis=0),slot['height_m']+obj.dimensions[2]/2]
                    placement = Placement(pose,slot['id'])
                    next_state = state|(1<<i)
                    next_cost = cost+float(np.linalg.norm(obj.pose[:3,3]-pose[:3,3]))
                    if next_cost < best.get(next_state,float('inf')):
                        best[next_state] = next_cost
                        heappush(frontier,(next_cost+.001*(len(blockers)-next_state.bit_count()),next_cost,next_state,(*path,(key,placement))))
    raise ValueError('support constraints leave no rearrangement order')


def prepare(scene, mode, options):
    context = dict(options)
    if mode == 'build_tower' and not context.get('base_id'):
        candidates = [o for o in scene.state.objects.values() if on_table(scene,o)]
        if not candidates:
            raise ValueError('no tower base observed')
        context['base_id'] = min(candidates,key=lambda o:o.pose[2,3]).id
    if mode in ('arrange_grid','build_color_pattern'):
        pattern = context.get('pattern')
        rows = len(pattern) if pattern else int(context.get('rows',2))
        columns = len(pattern[0]) if pattern else int(context.get('columns',2))
        if pattern and any(len(row)!=columns for row in pattern):
            raise ValueError('pattern must be rectangular')
        targets = grid_targets(scene,rows,columns,context.get('area'))
        context['assignments'] = assign_targets(scene,targets,None if pattern is None else [c for row in pattern for c in row])
    if mode == 'free_target':
        context['assignments'] = rearrangement_plan(scene,context['object_id'])
    return context


def next_step(scene, mode, context, excluded=()):
    if mode == 'build_tower':
        return stack_step(scene,int(context.get('count',2)),context.get('base_id'),excluded)
    if mode == 'unstack_tower':
        return unstack_step(scene,context.get('destination'),excluded)
    if mode in ('arrange_grid','build_color_pattern','free_target'):
        return assignment_step(scene,context['assignments'],excluded)
    if mode == 'reset_scene':
        if any(key not in scene.state.objects for key in scene.reset_targets):
            raise ValueError('reset requires re-observing missing original object IDs')
        return assignment_step(scene,[(key,p) for key,p in scene.reset_targets.items() if key in scene.state.objects],excluded)
    if mode == 'pack_box':
        from .tasks import next_transfer
        return next_transfer('collect_to_box',scene,destination=context.get('destination'),excluded=excluded)
    raise ValueError('unsupported experiment')


def execute(mode, scene, actions, refresh, check_cancel, **options):
    if mode == 'run_sequence':
        return run_sequence(options.get('sequence',DEFAULT_SEQUENCE),scene,actions,refresh,check_cancel,**{k:v for k,v in options.items() if k!='sequence'})
    refresh()
    context = prepare(scene,mode,options)
    results,deferred = [],set()
    for _ in range(max(1,len(scene.state.objects)*3)):
        check_cancel();refresh()
        step = next_step(scene,mode,context,deferred)
        if step is None:
            return results
        result = actions.relocate(*step)
        results.append(result)
        if not result.success:
            deferred.add(step[0])
            if result.evidence.get('critical') or (result.evidence.get('motion_started') and not result.evidence.get('recovered')):
                return results
    raise ValueError('task did not converge within bounded object transfers')


def sequence_steps(sequence, options):
    from .tasks import MODES as BASIC_MODES
    if not isinstance(sequence,(list,tuple)) or not sequence:
        raise ValueError('sequence must contain at least one task')
    steps = []
    for index,item in enumerate(sequence,1):
        if not isinstance(item,(str,dict)):
            raise ValueError(f'sequence stage {index}: expected a mode name or task settings')
        stage = item if isinstance(item,str) else item.get('mode')
        if not isinstance(stage,str) or stage=='run_sequence' or stage not in BASIC_MODES+MODES:
            raise ValueError(f'sequence stage {index}: unsupported mode')
        settings = dict(options) if isinstance(item,str) else options|{k:v for k,v in item.items() if k!='mode'}
        if stage in ('relocate','free_target') and not settings.get('object_id'):
            raise ValueError(f'sequence stage {index}: {stage} requires an object ID')
        if stage in ('relocate','stow','collect_to_box','pack_box') and not settings.get('destination'):
            raise ValueError(f'sequence stage {index}: {stage} requires a destination')
        steps.append((stage,settings))
    return steps


def run_sequence(sequence, scene, actions, refresh, check_cancel, **options):
    from .tasks import run_task
    steps = sequence_steps(sequence,options)
    results = []
    for stage,settings in steps:
        check_cancel();refresh()
        if stage in ('extract','empty_box') and not settings.get('destination'):
            stage_results = []
            for box_id in tuple(scene.state.boxes):
                check_cancel();refresh()
                part = run_task(stage,scene,actions,refresh,check_cancel,destination=box_id)
                stage_results.extend(part)
                if any(not r.success for r in part):
                    return results+stage_results
        elif stage in MODES and stage != 'run_sequence':
            stage_results = execute(stage,scene,actions,refresh,check_cancel,**settings)
        else:
            stage_results = run_task(stage,scene,actions,refresh,check_cancel,selected=settings.get('object_id'),destination=settings.get('destination'))
        results.extend(stage_results)
        if any(not r.success for r in stage_results):
            return results
    return results
