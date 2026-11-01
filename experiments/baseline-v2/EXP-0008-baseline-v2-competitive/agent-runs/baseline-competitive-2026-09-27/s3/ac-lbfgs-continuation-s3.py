"""Single-file Kaggle T4x2 launcher; worker and hash manifest are embedded."""
import base64
import datetime
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import zipfile

PAYLOAD_B64 = "UEsDBBQAAAAIAEh9PF0qzMaqKA8AAPIyAAASAAAAYWNfbGJmZ3Nfd29ya2VyLnB5xVttc+O2Ef7uX4Gg0xmyoRjL52SuSpQZ996aSeK7ufO103E9HFiEJNYUyRCkLcXVf+/uAiBBipJ1ybX1zEkUAewuFvvyYIHjnP8oFotUsqtz9pCXd7Jk8xz+JWsZj4o8T9lFmsps9EIsM/bT6C+v33xgszyrkqwWVZJnIef8ZF7mKxZF87qqSxlFLFkVeVkxkWV5Rb3UyYl9Vy4KUSppfy+FWqbJrf35L5Vn9nklqqV9zpV9UpvmsUpWUvMuoCtQsYzf4UhqqDZFki3s+4ts0wiS1atiw4RiWdHQy8vZ8uTk3cU/fnp78ZJNiY4H80pSmJUfllLl6b30/BCmILPq5MXby9c/vIGOKHaY5iJWnmeHf8W4ejZSeV3O5AhUNk8WIfbjSEjEUSXXlSezWR6DhFNeV/PRc+77Jxcvovdv315Z9prHNU+yoq6iMs8rxW+uucBViWawKvzGP/kwPmaMGmNfUGCI+gqTTMmy8k4DpqrSM2xBAKONskb6q0Ikiwz1dCuUZOwPLMt/ERP26vz07OTkJJZz9lAmlYxwah7SnZAMAbsXaS0nLE5m1TUwCFD7Nz4bfc8u80xOThj8kRxameHqLk5KT/9Q06uylgGT60RVUX5HP30aUkkUT5QbmCwNf0iqZZSJlSTuIT6xLxkPq1XBe0NCLSppnpYsBiNQHkkasCSLgfX0DPQB84/u5MaKAcrOH4BHNn0tUiV9pP/PjIN8/eXr8StlkYqZlsw36lJLcfb1N46qSCegIa2SOFlIVcHkjGeEpr8mjZPV084LmXm8vOU+Lg4Ml2KlKeAf+vBtms/uYFYM5lx6qVjdxrBwuieZoDc+PTtnf2L45QfslnO/pdDKEtZFLCrpET0tRinB0zPbvpRr/eTZOcp1IWeVjCNtgSiw8kS5IvagXynjCUhW0dSrukjltTYa/LzRQsyWcnZX5NALlGGdApzK2iSHH0ASPuccCY4e8XPLqU8zdjRPMpGGRcW1wWFE20uOY7PifZJhVvzK3Xm35AOiaGZ9L8tkvjFzxuWTx086cBwFPm52lBBY2ffpVnPQ69P06SjRhoX2pZESggMQuEHuHhG56VIxnO14/NmO7A0Ss6oWaZexMeH2ne92NdStX8Av3ZzMB4h9MR2aXGu2pUggTP0NPfpVWealN3emSz7FVomC1DJbkpM8wsS3pDlm7KfPnMRz2eKLgwxpxC6rXSZ7zClgj+4aacXwya42Am2xOkH1u2HL1lgmpiZ6oTohumuSrQFmRZjFoizFxpghhR14i3SIgo2JRTK7S6UNixSI8hKINuohprC6jxiXJ0iD6Hq63zW+vQHbhzwtp9A4Bw7VN+cBYIxio6M+6Y6iOoQyD3IaRLMkLyH68qVM47yuuL+1a0bsrttONxg+C4nr5z0f/xli+5nPgJ7pZwm43cbfPHt+jv0OrnGdWXtgStzDJy06UVF7VtvOAxcZBFQ4H5IjhAC9Up7DEGYC0Am1lSgIYNDu6TF+CHr3emF6QDwtEyl9q0VD0CYg5QPhbKRpwru8hPQFwX2fzFYYWLZsY2S4ngTs9IZ9x0bj8JS0OdT8PcPW3yAoOA5bd0SDFVJJLNn1aByw8c0nijpGUU/3STr+nZJWw5KeHhDUeD0tfQdCtY7d8dJVHst0otFpmGXhz3lcp2BFeQEAOPlVlraNXoRv7Wv2b0JaQWdizd9AYgqYzvTK/JKpKBS2kVciLEIsO+llqj30FSD/WhGLHuj7DQgOiUmMIg0vrulDuOPvXl2+/OHyTfTXjz9fXEYffnhzeXH18f0r3grGYbIYGCFJOgRg2pxmqtOX05TezhcqMsowfcwvt5vRT7Qo6khJcLBYRYCCEgj5khuteaaTO07rMRoK8brpeiD8O3oeCvp24E6TOw4hfaM3/RB0NSojXFxoNfG6khmA4TCWlZgtYeMzK2rPdyKZbsdIRlYatjSgswlrW82jidGN3bJEUZhrTaNZ62ve9IocuW7ABpqGDjNtWeQG6KgetQWtrR2DzPUmDbewRMjTQI1oxvI+mRkhTXDWDbM6FhCkI3EvklTcprA7xDjjNOqh0SyvwYp8TDHjfmZ5D02wl9WRhv9db8NXNWwDwDIBeEBGTzcMtAQZVgFWhYjy4uPLC6ZJf8tevPvI5pAYbgUCfgUOqlCUOOSuXkgaJatID/NOdSO529Tts2j6aK88bSARr845TR4WHJsAEDzA3sI/OKG5nVEpf6kT2EQzkbUTuYJcOwfdmND6RbntCI3ywlY9qpa4Y1HeWb8RFk7UKXwjhPD0e4Mi3K6oGgkOqmcIHrqq01BjmGr+7AwUQBhmzwiIunv6mmDuWonHkcfklFu7qkApmTcYcCFhAIi3kX4mChM2yPQaYKvpAt8dC+1tEiKNzSgI6EcTazRaBwp79iju5sFCNhc0OgSbfh2IrxVA+HBHnJUoItg7UiVoyiGEAHZ7kMliWakoz9KNgY/WyNrxaIkeBW/yG9ztUbCpvH4fCuYBIBKfepKGD8E3rRZ3Cglu/pNqc9z2gKIdTBuLIuFtnaRxRK8cXYZV7umFm+ovC3MHbFQHT1K4E9Ra6a7d4HxDeReenKqIJqANza4iYUy7MkJFOlZ7OxB5UK6AdWTXNLW1Hq7UOGsJPcCjgQ1ZE7R7Wsz2vecbwk1KmHawzE9Yb/TazUQzsE1baTkF+Nb+Xol1hAWP6bj7Tt6LdPq8fVflqSxFBr6xKEWMNORosHm2FNlCTk9dJssE9y+bSIHM0/Gp05ImGaReKWAO0RysHVYqzxbRQ57OpUEkDaQpcfcwpTJmWMhyrnMExlPqYRAH9DjV/pbmCl0fZ6KLr21bKkD70K5M/DDYD9rxS0eqJvnDW4QCWZItdFlDoldQcBoYRwUU8mRMvmQBh0ssZb6AIK92h9gWXQY9MaFtbnGmzrck/sRZhzZ1bbLZErQJOvdcq3TCMCy0hws1pFM2sjr3W9aIFSIrl7cfs9KKaYiHJRgrcNO2A+BbrRkA7+D1gDVRooHcgSXfC9mBwWl9VlRu7SDvjt67oPxxB6LzvTDwEGbuY2eawW5rHz7bKe72HDBp6D/wdoiLNflIzQSAnWilEHU3rweGPAHY7QrsDnQHQJJGGUUxpJenkL1Z1ScA/tNAv6FzCO+3Ys0STG7zZBaJ2UwWFa0eRepu72370zhJ10E4pqhO5c2CtXLTOgphab2GLoa+ovzT3WdnkP8BGaSDa96uZWdMuwGAfzmFbw+RYJVHQE46WbGNDrArr0WTsItYRvadZ7zT1Fc645A1jLFdQ/VLLXBO4UqKzOv27ewKmqoNUuiVFlpE8hpDHQThd8jagH+nOjNw8KaVRHLxXVEJsj6IMt4VjYoeIarK3XTB63hIat3T1JoIBxVUrGrS7+ebEjJC4+xNZyjXfQk7p+6crXnAGunNNmmhu1mlXajFGb10QabVBtYluJGTjL4jPL7H9FQlC89I2aVt0/aOuLAMtu2P7Gs2hdy9q8aev5k0HVFIdXTUTeaY7TAugVHDzMG5TfkeXZ29oi/QIJZn4d1kDxEdtaXtzZtuhA5QxfyR9ljQxQ8j2hpG0XbCHuHFlu+Gge5UWmb+cdJZtp5++JLxbxlHJerfEoIX4xzP4uZcZ1snplKf6ZPy7mrBoaET7Rz29VajQ6nXQUS5qjS0aaNoN/O2Wbdl2g3A+7Lu/ox7XLb9rZn2E7OsEeaIdMlpr4KmvScZN37YG+dCb+jWAd/UwYL9qKDakUT2z/tidlA6FhEHcPp/I6t/hozOybZRQ/jdF/XYTM9LOZewgQPG+ztu3Q2LiZljdEHHZyAvPO7EjmC/I2215566J1ONLP1T8c9yJIWrnNw7QH4dme3/tD2PMp2u+Zo/cR7Vbk0OkamOJjNPZBoP0qiPogHLQSTMERakFQ+SrGcn6QOMgp9WWN9J3ZazHn41RO0LQ63qUlsPUBuosjTralhpmo8Ogy2Lc6kBiS66JJVi668qwAVJbBOe9metIvBcVWDhFg+fAkYfX4/PdvVkcOnQ0NNmJLjXvpEixXiw1kPB1md33jWex+FmsvC0SEFjSgEr8wcNleABnYImCesn1omamgqqCbMy3kO1MlSrhqqRAk+q8HrK+kbzoGfkUmJtwqw3jvV9y3Lc8dpHsOsJs1JzrPBbZmBmk1Yye2KsZjkg3cGaJTp/nJS2aOmWMP+fZUssyPVIflI5Uz3rXlow0zxY6zCgvzMUTwMwfO9WxV/D28u8eo3Vbusilzn78Kxzo84tSj5Zgowg14tu+bUjzKcVX/8nVc1W7iOrmmib3rFFM7Jb3NzVaaX6t8/wBHHbnMVDdIraewWvX+L9AMxcr65evv/x3L0LhT1NNctrrw7Z6zBtJqW7OPzm2lIGb8VkaflAaLZsdBL07L2944i10NhePZgOjGsu5jQDm3H6Zkh718bOq3OAbjo5N16ePBp/tKy2rA35nQsw7tal6THtZ/9deTrL2c7I2qgNWxFFW7uVbwjaIrYNhd0qgLn8MVAJ6N4N6VJpq4AHa4yWI8i979ya5hYfeUzdBf6fC5L+TjjK3bg1LIw2tE5Mco/CS4D5FYCdSIck2F9Qio5oR5ueIQXXApyRdvXcWk5UrlSU5eVKpLA7QLXZFld1QBD3HQ8AF+3GY6j4p9cnQiCCqwUgYw1T0Tcrc4gF8zL/VWatsSneGXsEDn8Sg2/7u06dawNjWJ0MbwH1io6AOnmYLlzjVtpevg4vykW9AgHfUYvnO91CEceRMO0eH43QKyAszpY5+ICaenrnhoGSFMT9wB7sxk7o3kMMLbylRZcgPXLGi/c/f/gkSubEj3IPJbouURsXsRsEw08ijV5vKGuIc/xQAA0jAA3u8IP9cXds+ppLNuY8e3p2jneWNBQtF4ruyhAJ+kIiyhuovWIAh6YQV40Sjl6tyVD5y55LQ3dK8PSkTyXoMUdToycQ099hYQFSonoHJc58pS4BNorBzlab+nqU9ka+c5RjAeg+4Qz3VtQjy0q2+ydeOz908tIjCdnc4/r/Uoxw0wu4V5eF9i2OqWXRjLsjesWPRyd7mO00uCE56KSlHLSJxCrPZpFWiSdssKZwuMYXHAxsncOD9lSKTsO4/b8lJDXQO8gF080U/5+CqmKQbMc2xhDrQJd2JKkyijDyRZExdY1SPmxUJVev1gnAToqL/sl/AFBLAwQUAAAACABIfTxdy4yteTYFAAAgEQAAFQAAAHMzLXNvdXJjZS1jb25maWcuanNvbsVXTW8bNxC9B8h/MHSOoiE5/Jje2jq5tAWKpIcCRSEMyaEtWNYKWjkoGuS/d1a2JVlS6wa1XcCAKe5y3sx7b8jl59evzs5GvfT9rFuMvjkbfXSjN3dzUnud+W34dXaGYN7cj+x25LYj3I78MPj9NgivrvdijL79fvzhu/fjj13u5qM3u8lzkeWv5+/GP/3482hvdW942q95fTPEGBVe1FnltUxLd72cy1ruMp0tljfr6arr1sN7n+/C8nwui2nhy01Zkyu+uJjLZPPuZNGtJXfdVT9ZdXU1u9bIN3nCbbycXfGiH2/Wjoe14zX+YccWbACycXLwYFtDb74eZKhC1rP17JOMM/cyny1k3Jt7sDQp866/WQ1zG1K+3Fa7kiYrWRSZLnl9uV/x+/NkyN5Sdb3k2cVisn15wmV6sbyZ7lZv6Xy7WP65LeTdL+cffsAhxo6/iazr6gp3azcr9lK67qrMp313s9K4/SVbH4YIGAvYorSR+IDRUhORhKlhIxsCRwzMhlKgaGMtGWMNyXIFNuAsmjt5y6WUq2U3W6ynl9xfyn7JDwy1ndYHatchBUiOi2nGUNUUSmtRXHS2NioSQ0LxMTY0LLZagxWgBHYQSjI2O+vonpdNxA21TFBK4qg1egoWxAWvryZKuSYP3MA6IVYwtsZxSlFSIsPso2BifBDRDRGjL9BMbdV7CZYLmBijNl82ponxmTwRsAsZlCFvYyrVttxK1Tx8cP5BxI10GThkSiSl1ZyjGxZaG6wF8qLcm1BaClbTJUfFNksBWks6SKQ/3IOIfohIKqVWJDmAtOxzc+xaTsqXabEov06p9TU3H1Tj6KNWDCqtr7EY9cDoNuCXv+n5E8qxoPXeSkCUyqqQZlxM9ibmyq4SRAqmoXINkXNQTKDQOKoolInysXJZcsk1kNTqTS6IkElMdcGUYKqWJkpxiSDReJVCSHV1vrSIBbgEOVbOa5xam5IrUDUf5dD5RoKk9OhaGUQomrxHV3JOsTpXdIwhJWkGj5UDG5U45c1IAmRGcdoLWHKowTUbsthApM2krZTU14qnMXPLFUN1UI+Vy94SNDWAUy9JaEgMqaIXY2PmHIOqiN5RjGBcTawka2c0ZT8nI0buldtr92XXzY9b8b7jQnRkS8waLagfaksUsqcMLYATn1zMhIFQdwUTvG9RORZqLhOoI9N2J7pTzRtjPHHTMrU+ozajxpRAAKy6T/vAN6jaeOwhmqraGc+lBf0XoRLtxbvVzHmbYyW1bMBa9Z0cUWUPoSoFefBaIl+qoIrumvHIarGhnZw1gnkvHt52hg2xxaRLPDtPqSZtT6dGlMA2SA2ai1h0GgkhaiOXUrQJK2kT6c+9eBu9bCtQLZH6Bq26WneOjAIGiFKBHFR1lV0ZKNGnJmxTIaFhOkdrKJ4+Ko7U2p0VaWA/69YAYlNgVcRoy0F11ifIgLWAV1toi0RuHvQRe93bku5J6kVwJ46OBurdFkQrodQasa5BY7XX2rC/C9ZMFUwD0r5wSClqH2E0yWbkQrhfw97uP+8Kz49OvceOgNszedzdrPVQ7g9P8PGnvQP34Nlke4zuY0yGD6OxBp/sUhu32YLnb5fr413nueDto/DuOeHdo/D4nPD4KLx/Tnh/Ev6rz7gnSm8P5f8x54kEXtaeJxJ4WYOeSOBlLXoigX806eFxfnpvfUKfDij91p4P7hxPaMYHKPYAxT0LijtAwWdBwQMU/ywo/vBmt/uCOG2Q3WfEU+TyH26rB/C7q/Op6/TkX91tX7/Sv78AUEsBAhQAFAAAAAgASH08XSrMxqooDwAA8jIAABIAAAAAAAAAAAAAAIABAAAAAGFjX2xiZmdzX3dvcmtlci5weVBLAQIUABQAAAAIAEh9PF3LjK15NgUAACARAAAVAAAAAAAAAAAAAACAAVgPAABzMy1zb3VyY2UtY29uZmlnLmpzb25QSwUGAAAAAAIAAgCDAAAAwRQAAAAA"
KERNEL_SLUG = "rodrimandub/af-pikans-ac-lbfgs-continuation-s3-retry-20260928"
SESSION_START = time.monotonic()
SESSION_STARTED_UTC = datetime.datetime.now(datetime.timezone.utc).isoformat()
SESSION_CAP_SECONDS = 2674.641350559
S3_STAGE_TOTAL_CAP_SECONDS = 2700.0
PREVIOUS_S3_ATTEMPT_SECONDS = 25.358649441
SESSION_CLOSE_MARGIN_SECONDS = 60.0
RUN_CAP_SECONDS = 240.0
TRAIN_WAVE_WALL_SECONDS = 270.0
SCORE_WAVE_WALL_SECONDS = 240.0
SEEDS = [401, 402, 403, 404, 405]
ARMS = ["AC-RBF-Sobol", "AC-DeepXDE-MLP"]
KERNEL_SOURCES = [
    "rodrimandub/af-pikans-allen-cahn-t4x2-20260927",
    "rodrimandub/af-pikans-competitive-baseline-s1-20260928",
]

WORK = pathlib.Path("/kaggle/working/closure-s3-20260928")
WORK.mkdir(parents=True, exist_ok=True)
PAYLOAD_DIR = WORK / "payload"
PAYLOAD_DIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD_B64))) as bundle:
    bundle.extractall(PAYLOAD_DIR)
RUNNER = PAYLOAD_DIR / "ac_lbfgs_worker.py"
CONFIG_PATH = PAYLOAD_DIR / "s3-source-config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
AC_ROOT = pathlib.Path(CONFIG["input_roots"]["allen_cahn"])
S1_ROOT = pathlib.Path(CONFIG["input_roots"]["s1"])
LEDGER_PATH = WORK / "campaign-ledger.json"
LOGS = WORK / "worker-logs"
LOGS.mkdir(parents=True, exist_ok=True)
LEDGER = {
    "campaign": "allen-cahn-lbfgs-continuation-s3-retry-2026-09-28",
    "status": "starting",
    "kernel_slug": KERNEL_SLUG,
    "session_cap_seconds": SESSION_CAP_SECONDS,
    "per_arm_seed_cap_seconds": RUN_CAP_SECONDS,
    "accelerator": "NvidiaTeslaT4 (GPU T4 x2)",
    "seeds": SEEDS,
    "arms": ARMS,
    "scientific_accepted": False,
    "reference_accepted": False,
    "source_preflight": {},
    "runs": [],
}


def elapsed():
    return max(0.0, time.monotonic() - SESSION_START)


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_ledger(status=None):
    if status:
        LEDGER["status"] = status
    LEDGER["session_wall_seconds"] = elapsed()
    write_json_atomic(LEDGER_PATH, LEDGER)
    manifest = {
        "schema_version": 1,
        "session": "S3",
        "kernel_slug": KERNEL_SLUG,
        "status": LEDGER["status"],
        "started_at_utc": SESSION_STARTED_UTC,
        "session_wall_seconds": elapsed(),
        "session_cap_seconds": SESSION_CAP_SECONDS,
        "session_quota_cap_hours": SESSION_CAP_SECONDS / 3600.0,
        "s3_stage_total_cap_seconds": S3_STAGE_TOTAL_CAP_SECONDS,
        "previous_s3_attempt_seconds": PREVIOUS_S3_ATTEMPT_SECONDS,
        "per_arm_seed_cap_seconds": RUN_CAP_SECONDS,
        "runtime": LEDGER.get("runtime"),
        "source_preflight": LEDGER.get("source_preflight"),
        "runs": LEDGER["runs"],
        "scientific_accepted": False,
        "reference_accepted": False,
    }
    write_json_atomic(WORK / "manifest.json", manifest)


def require_t4x2():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("S3 requires exactly two visible T4 devices; no fallback is allowed.")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("t4" not in name.lower() for name in names):
        raise RuntimeError(f"S3 requires T4x2; visible devices={names}")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu_names": names, "visible_gpu_count": torch.cuda.device_count()}


def source_path(kind, arm=None, seed=None):
    if kind == "checkpoint":
        return AC_ROOT / "campaign" / arm / f"seed-{seed}" / "checkpoint-final.pt"
    if kind == "pool":
        return AC_ROOT / "campaign" / "pools" / f"seed-{seed}.npz"
    if kind == "FD8192":
        return AC_ROOT / CONFIG["reference_paths"][kind]
    if kind == "ETDRK4":
        return S1_ROOT / CONFIG["reference_paths"][kind]
    if kind == "model_source":
        return AC_ROOT / "run_campaign.py"
    raise ValueError(f"Unknown preflight artifact {kind}")


def verify_sources():
    observed = {"model_source": sha256(source_path("model_source")), "checkpoints": {}, "pools": {}, "references": {}}
    if observed["model_source"] != CONFIG["model_source_sha256"]:
        raise ValueError("Allen-Cahn model source hash mismatch")
    for seed in SEEDS:
        pool_file = source_path("pool", seed=seed)
        actual_pool = sha256(pool_file)
        if actual_pool != CONFIG["pool_hashes"][str(seed)]:
            raise ValueError(f"saved pool hash mismatch for seed {seed}")
        observed["pools"][str(seed)] = actual_pool
        observed["checkpoints"][str(seed)] = {}
        for arm in ARMS:
            checkpoint_file = source_path("checkpoint", arm=arm, seed=seed)
            actual_checkpoint = sha256(checkpoint_file)
            expected = CONFIG["checkpoint_hashes"][arm][str(seed)]
            if actual_checkpoint != expected:
                raise ValueError(f"checkpoint hash mismatch for {arm} seed {seed}")
            observed["checkpoints"][str(seed)][arm] = actual_checkpoint
    for ref in ("FD8192", "ETDRK4"):
        actual = sha256(source_path(ref))
        if actual != CONFIG["reference_hashes"][ref]:
            raise ValueError(f"{ref} reference hash mismatch")
        observed["references"][ref] = actual
    observed["s1a_status"] = CONFIG["s1a_status"]
    observed["all_hashes_verified"] = True
    return observed


def launch_pair(mode, seed):
    if mode not in {"train", "score"}:
        raise ValueError(mode)
    cap_wall = TRAIN_WAVE_WALL_SECONDS if mode == "train" else SCORE_WAVE_WALL_SECONDS
    if elapsed() + cap_wall + SESSION_CLOSE_MARGIN_SECONDS > SESSION_CAP_SECONDS:
        save_ledger(f"stopped_session_guard_before_{mode}_seed_{seed}")
        return False
    processes = []
    wave_start = time.monotonic()
    for device_index, arm in enumerate(ARMS):
        run_dir = WORK / "runs" / arm / f"seed-{seed}"
        if mode == "train":
            if run_dir.exists():
                raise FileExistsError(f"Refusing to reuse S3 run directory: {run_dir}")
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            out_path = run_dir
            args = [sys.executable, str(RUNNER), "--mode", "train", "--arm", arm,
                    "--seed", str(seed), "--out", str(out_path), "--cap", str(RUN_CAP_SECONDS)]
        else:
            score_path = run_dir / "score.json"
            if not (run_dir / "checkpoint-final.pt").is_file():
                LEDGER["runs"].append({"mode": "score", "arm": arm, "seed": seed,
                                       "status": "missing_continuation_checkpoint"})
                continue
            args = [sys.executable, str(RUNNER), "--mode", "score", "--arm", arm,
                    "--seed", str(seed), "--run-dir", str(run_dir), "--out", str(score_path)]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(device_index)
        log_path = LOGS / f"{mode}-seed-{seed}-{arm}.log"
        stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(args, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        processes.append((arm, seed, run_dir, process, stream, log_path))

    while any(item[3].poll() is None for item in processes):
        if elapsed() >= SESSION_CAP_SECONDS - SESSION_CLOSE_MARGIN_SECONDS or time.monotonic() - wave_start >= cap_wall:
            for item in processes:
                if item[3].poll() is None:
                    item[3].terminate()
            for item in processes:
                try:
                    item[3].wait(timeout=10)
                except subprocess.TimeoutExpired:
                    item[3].kill()
                    item[3].wait(timeout=10)
            save_ledger(f"stopped_{mode}_wave_guard_partial")
            break
        time.sleep(1.0)

    for arm, seed, run_dir, process, stream, log_path in processes:
        return_code = process.poll()
        if return_code is None:
            return_code = process.wait()
        stream.close()
        row = {"mode": mode, "arm": arm, "seed": seed, "return_code": return_code,
               "log": str(log_path.relative_to(WORK))}
        if mode == "train":
            cost_path = run_dir / "cost.json"
            progress_path = run_dir / "progress.json"
            if cost_path.is_file():
                cost = json.loads(cost_path.read_text(encoding="utf-8"))
                row.update({"status": cost.get("status"), "lbfgs_updates": cost.get("lbfgs_updates"),
                            "closure_evaluations": cost.get("closure_evaluations"),
                            "training_gpu_seconds": cost.get("training_gpu_seconds_estimate"),
                            "continuation_checkpoint": str((run_dir / "checkpoint-final.pt").relative_to(WORK))
                            if (run_dir / "checkpoint-final.pt").is_file() else None})
            elif progress_path.is_file():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                row.update({"status": progress.get("status"), "lbfgs_updates": progress.get("lbfgs_updates"),
                            "partial_progress": str(progress_path.relative_to(WORK))})
            else:
                row["status"] = "worker_failed_before_progress"
        else:
            score_path = run_dir / "score.json"
            if score_path.is_file():
                score = json.loads(score_path.read_text(encoding="utf-8"))
                row.update({"status": score.get("status"), "scoring_wall_seconds": score.get("scoring_wall_seconds"),
                            "score": str(score_path.relative_to(WORK))})
            else:
                row["status"] = "scorer_failed"
        LEDGER["runs"].append(row)
        save_ledger()
    return True


def main():
    save_ledger("starting_preflight")
    try:
        LEDGER["runtime"] = require_t4x2()
        LEDGER["source_preflight"] = verify_sources()
        save_ledger("sources_verified")
    except Exception as exc:
        LEDGER["preflight_error"] = f"{type(exc).__name__}: {exc}"
        save_ledger("preflight_failed")
        raise

    for seed in SEEDS:
        if not launch_pair("train", seed):
            break
    trained = [row for row in LEDGER["runs"] if row.get("mode") == "train"]
    save_ledger("training_stage_finished" if len(trained) == len(SEEDS) * len(ARMS) else "training_partial")

    if elapsed() < SESSION_CAP_SECONDS - SESSION_CLOSE_MARGIN_SECONDS:
        for seed in SEEDS:
            if not launch_pair("score", seed):
                break

    scored = [row for row in LEDGER["runs"] if row.get("mode") == "score" and row.get("status") == "scored"]
    training_complete = len(trained) == 10 and all(row.get("status") == "time_cap_reached" for row in trained)
    status = "complete" if training_complete and len(scored) == 10 else "partial"
    LEDGER["completion"] = {
        "training_runs": len(trained),
        "training_reaching_time_cap": sum(row.get("status") == "time_cap_reached" for row in trained),
        "scored_runs": len(scored),
        "scientific_accepted": False,
        "reference_accepted": False,
    }
    LEDGER["ended_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save_ledger(status)
    artifacts = []
    for path in sorted(WORK.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append({"path": str(path.relative_to(WORK)), "size_bytes": path.stat().st_size,
                              "sha256": sha256(path)})
    final_manifest = {
        "schema_version": 1,
        "session": "S3",
        "kernel_slug": KERNEL_SLUG,
        "status": status,
        "started_at_utc": SESSION_STARTED_UTC,
        "ended_at_utc": LEDGER["ended_at_utc"],
        "actual_session_seconds": elapsed(),
        "push_cap_seconds": SESSION_CAP_SECONDS,
        "session_quota_cap_hours": SESSION_CAP_SECONDS / 3600.0,
        "s3_stage_total_cap_seconds": S3_STAGE_TOTAL_CAP_SECONDS,
        "previous_s3_attempt_seconds": PREVIOUS_S3_ATTEMPT_SECONDS,
        "per_arm_seed_cap_seconds": RUN_CAP_SECONDS,
        "runtime": LEDGER.get("runtime"),
        "source_preflight": LEDGER.get("source_preflight"),
        "arms": ARMS,
        "seeds": SEEDS,
        "optimizer": {"name": "PyTorch LBFGS", "history_size": 100,
                      "tolerance_grad": 1.0e-8, "tolerance_change": 0.0,
                      "line_search_fn": "strong_wolfe", "per_step_max_iter": 1,
                      "per_step_max_eval": 8},
        "references": ["FD8192 candidate", "independent ETDRK4 candidate"],
        "score_grid": "512x256",
        "kernel_sources": KERNEL_SOURCES,
        "artifacts": artifacts,
        "scientific_accepted": False,
        "reference_accepted": False,
    }
    write_json_atomic(WORK / "manifest.json", final_manifest)
    print(json.dumps({"status": status, "session_seconds": elapsed(),
                      "training_runs": len(trained), "scored_runs": len(scored),
                      "scientific_accepted": False}, indent=2))


main()
