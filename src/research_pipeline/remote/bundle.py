"""Build explicit, reviewable source bundles for Kaggle and Colab."""
from __future__ import annotations

import json
import re
import ast
import importlib.util
from pathlib import Path

from src.research_pipeline.pipeline import load_yaml, validate_manifest
from .runtime import canonical, contained, digest, verify_bundle, write_json

ROOT = Path(__file__).resolve().parents[3]


def validate_local_imports(content):
    """Reject source bundles that omit imported modules from this repository."""
    included = {Path(name.replace('\\', '/')).as_posix() for name in content if name.endswith('.py')}
    missing = set()

    def check_module(module):
        if not module or not module.split('.', 1)[0] in {'src', 'evals'}:
            return
        parts = module.split('.')
        for index in range(1, len(parts)):
            parent_init = ROOT.joinpath(*parts[:index], '__init__.py')
            if parent_init.is_file():
                relative = parent_init.relative_to(ROOT).as_posix()
                if relative not in included:
                    missing.add(relative)
        base = ROOT.joinpath(*parts)
        package_init = base / '__init__.py'
        module_file = base.with_suffix('.py')
        source = package_init if package_init.is_file() else module_file
        if source.is_file():
            relative = source.relative_to(ROOT).as_posix()
            if relative not in included:
                missing.add(relative)

    for name, data in content.items():
        if not name.endswith('.py'):
            continue
        source_path = Path(name.replace('\\', '/'))
        parts = list(source_path.with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
            module = '.'.join(parts)
            package = module
        else:
            module = '.'.join(parts)
            package = module.rpartition('.')[0]
        tree = ast.parse(data.decode('utf-8'), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    check_module(alias.name)
            elif isinstance(node, ast.ImportFrom):
                target = node.module or ''
                if node.level:
                    if not package:
                        continue
                    target = importlib.util.resolve_name('.' * node.level + target, package)
                check_module(target)
                for alias in node.names:
                    if alias.name != '*':
                        check_module(f'{target}.{alias.name}' if target else alias.name)

    if missing:
        raise ValueError('Bundle source imports are not fully packaged: ' + ', '.join(sorted(missing)))


def repository_path(path):
    target = Path(path).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError('Path must stay inside the repository: ' + str(path))
    return target


def generated_path(path):
    target = repository_path(path)
    if not target.is_relative_to(ROOT / 'var' / 'agent_runs'):
        raise ValueError('Generated output must be under repository var/agent_runs')
    return target


def render_notebook(spec, content, manifest):
    from .notebook import render_notebook as render_structured_notebook
    return render_structured_notebook(spec, content, manifest)


def render_metadata(spec):
    kernel = spec['kernel']
    metadata = {'id': kernel, 'title': kernel.split('/')[1].replace('-', ' '), 'code_file': 'experiment.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': spec['enable_gpu'], 'enable_internet': spec.get('enable_internet',False),
                'dataset_sources': [], 'competition_sources': [], 'kernel_sources': [], 'model_sources': []}
    if spec.get('machine_shape'):
        metadata['machine_shape'] = spec['machine_shape']
    return metadata


def prepare(manifest_path, output, kernel, files, timeout=300, evidence_kind='research', gpu=False):
    output = generated_path(output)
    if output.exists():
        raise ValueError('Output already exists; use a new bundle directory')
    manifest_path = repository_path(manifest_path)
    manifest = load_yaml(manifest_path)
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError('; '.join(errors))
    if not re.fullmatch(r'[a-zA-Z0-9_-]+/[a-z0-9]+(?:-[a-z0-9]+)*', kernel):
        raise ValueError('Expected kernel owner/lowercase-slug')
    seeds = manifest.get('training', {}).get('seeds')
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 32 or
            any(type(s) is not int or not 0 <= s < 2**32 for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError('seeds must contain 1..32 unique unsigned 32-bit integers')
    if not 0 < timeout <= 3600:
        raise ValueError('timeout must be between 1 and 3600 seconds per seed')
    if evidence_kind not in ('research', 'infrastructure'):
        raise ValueError('Unknown evidence kind')
    content = {}
    for name in files:
        path = contained(ROOT, name)
        if not str(name).startswith(('src/','evals/')) or path.suffix != '.py':
            raise ValueError('Only explicitly listed src/*.py or evals/*.py source files can be packaged')
        content[str(name)] = path.read_bytes()
    entrypoint = manifest['model'].get('implementation')
    if entrypoint not in content:
        raise ValueError('model.implementation must be included with --file')
    if evidence_kind == 'research' and ('stub' in entrypoint or 'infra_smoke' in entrypoint):
        raise ValueError('Stub/smoke code cannot produce research evidence')
    required = sorted(set(manifest.get('artifacts', {}).get('expected', [])) |
                      {'metrics.json', 'decision_note.md'})
    # run_summary is generated after artifacts have been collected.
    required = [name for name in required if name != 'run_summary.json']
    for name in required:
        contained(output, name)
    content['manifest.json'] = canonical(manifest)
    content['runtime.py'] = Path(__file__).with_name('runtime.py').read_bytes()
    content['src/research_pipeline/remote/postprocess.py'] = Path(__file__).with_name('postprocess.py').read_bytes()
    validate_local_imports(content)
    spec = {'schema_version': 1, 'kernel': kernel, 'entrypoint': entrypoint, 'seeds': seeds,
            'timeout_sec': timeout, 'max_total_training_sec': len(seeds) * timeout,
            'evidence_kind': evidence_kind, 'enable_gpu': gpu, 'required_artifacts': required,
            'files': {name: digest(data) for name, data in sorted(content.items())}}
    remote=manifest.get('remote',{})
    dependencies=remote.get('runtime_dependencies',[])
    if not isinstance(dependencies,list) or any(not isinstance(d,str) or
            not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*==[A-Za-z0-9][A-Za-z0-9_.+!-]*',d) for d in dependencies):
        raise ValueError('Remote dependencies must be exact name==version pins')
    internet=remote.get('enable_internet',False)
    if type(internet) is not bool: raise ValueError('enable_internet must be boolean')
    if dependencies and not internet: raise ValueError('Pinned install requires explicit enable_internet')
    parallel=remote.get('max_parallel_seeds',1)
    if type(parallel) is not int or not 1<=parallel<=min(8,len(seeds)):
        raise ValueError('max_parallel_seeds must be an integer from 1 to min(8, seed count)')
    machine_shape=remote.get('machine_shape')
    if machine_shape is not None and machine_shape!='NvidiaTeslaT4':
        raise ValueError('Unsupported explicit machine_shape')
    if parallel>1 and (not gpu or machine_shape!='NvidiaTeslaT4'):
        raise ValueError('Parallel seed execution requires GPU mode and an explicit T4x2 machine_shape')
    if 'allow_gpu_count_fallback' in remote:
        if type(remote['allow_gpu_count_fallback']) is not bool:
            raise ValueError('allow_gpu_count_fallback must be boolean')
        spec['allow_gpu_count_fallback'] = remote['allow_gpu_count_fallback']
    if dependencies:
        spec['runtime_dependencies']=dependencies
        setup_timeout=remote.get('setup_timeout_sec',900)
        if type(setup_timeout) is not int or not 1<=setup_timeout<=900:
            raise ValueError('setup_timeout_sec must be an integer in 1..900')
        spec['setup_timeout_sec']=setup_timeout
    if internet: spec['enable_internet']=True
    spec['max_parallel_seeds']=parallel
    if machine_shape: spec['machine_shape']=machine_shape
    # Bind the generated notebook logic to the bundle approval. The notebook
    # embeds the bundle ID, so hashing the notebook itself here would be circular;
    # hashing its deterministic renderer captures the editable transport logic.
    spec['notebook_template_sha256'] = digest(Path(__file__).with_name('notebook.py').read_bytes())
    spec['bundle_id'] = digest(canonical(spec))
    content['bundle.json'] = canonical(spec)
    # Notebook is self-contained: no Git clone, credentials or implicit repo upload.
    nb = render_notebook(spec, content, manifest)
    output.mkdir(parents=True)
    for name, data in content.items():
        target = contained(output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    write_json(output / 'experiment.ipynb', nb)
    write_json(output / 'kernel-metadata.json', render_metadata(spec))
    write_json(output / 'transport-hashes.json', {
        name: digest((output / name).read_bytes()) for name in ['experiment.ipynb', 'kernel-metadata.json']})
    return spec


def verify_transport(bundle):
    bundle = Path(bundle)
    spec = verify_bundle(bundle)
    template_hash = spec.get('notebook_template_sha256')
    if template_hash is not None and template_hash != digest(Path(__file__).with_name('notebook.py').read_bytes()):
        raise ValueError('Notebook template hash mismatch; the approved transport renderer has changed')
    from .runtime import read_json
    index = read_json(bundle / 'transport-hashes.json')
    if set(index) != {'experiment.ipynb', 'kernel-metadata.json'}:
        raise ValueError('Transport hash index must contain exactly notebook and metadata')
    for name, expected in index.items():
        if digest(contained(bundle, name).read_bytes()) != expected:
            raise ValueError('Transport hash mismatch: ' + name)
    content = {name: contained(bundle, name).read_bytes() for name in spec['files']}
    content['bundle.json'] = canonical(spec)
    notebook = render_notebook(spec, content, read_json(bundle / 'manifest.json'))
    if read_json(bundle / 'experiment.ipynb') != notebook:
        raise ValueError('Transport notebook does not match the approved bundle')
    if read_json(bundle / 'kernel-metadata.json') != render_metadata(spec):
        raise ValueError('Transport metadata does not match the approved bundle')
    return spec
