"""Save verified public assets for reuse as a private Kaggle notebook input."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def allowed(name):
    path=Path(name)
    return (not path.is_absolute() and '..' not in path.parts and
            (name.startswith(('rag_study/hf_cache/', 'rag_study/data/pdfs/',
                              'rag_study/data/index/', 'ollama-models/')) or
             name in ('ollama-linux-amd64.tar.zst', 'rag_study/data/index_signature.json')))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def publish_assets(base, destination, manifest):
    base,destination=Path(base),Path(destination)
    names=set(manifest.get('remote_files',{}))
    for relative in ['ollama-models','rag_study/data/index']:
        names.update(str(p.relative_to(base)) for p in (base/relative).rglob('*') if p.is_file())
    names.update(n for n in ['ollama-linux-amd64.tar.zst','rag_study/data/index_signature.json'] if (base/n).is_file())
    if not all(allowed(n) and (base/n).resolve().is_relative_to(base.resolve()) for n in names):
        raise ValueError('Cache may contain public assets only, never source code or credentials.')
    destination.mkdir(parents=True,exist_ok=True)
    record={'scope':'public assets only; private notebook output; preserve upstream licences',
            'sha256':{n:digest(base/n) for n in sorted(names)}}
    temporary=destination/'shared-assets.tmp'
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(names):archive.write(base/name,name)
    temporary.replace(destination/'shared-assets.zip')
    (destination/'shared-assets.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


def restore_assets(cache, base):
    cache,base=Path(cache),Path(base)
    record=json.loads((cache/'shared-assets.json').read_text())
    with zipfile.ZipFile(cache/'shared-assets.zip') as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(record['sha256']) or not all(allowed(n) for n in names):
            raise ValueError('Invalid asset-cache file list.')
        for name in names:
            target=base/name
            if not target.resolve().is_relative_to(base.resolve()):raise ValueError('Unsafe cache path.')
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.restore')
            with archive.open(name) as source,temporary.open('wb') as sink:
                shutil.copyfileobj(source,sink,1024*1024)
            if digest(temporary)!=record['sha256'][name]:
                temporary.unlink()
                raise ValueError('Asset cache hash mismatch: '+name)
            temporary.replace(target)
    return len(names)
