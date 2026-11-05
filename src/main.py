import os
import resource


def raise_nofile_limit() -> None:
    """
    Raise this process's open-file limit (RLIMIT_NOFILE) at startup.
    Same effect as: prlimit --nofile=65536:524288 --pid 1

    The soft limit can always be raised up to the hard limit. Raising the
    hard limit itself needs CAP_SYS_RESOURCE; if that is not allowed we keep
    the current hard limit and still raise the soft limit as far as it goes.
    """
    want_soft = int(os.getenv("NOFILE_SOFT_LIMIT", "65536"))
    want_hard = int(os.getenv("NOFILE_HARD_LIMIT", "524288"))
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"[main] open-file limit before: soft={soft} hard={hard}")

        if hard != resource.RLIM_INFINITY and hard < want_hard:
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, want_hard), want_hard))
                hard = want_hard
            except (ValueError, OSError) as exc:
                print(f"[main] could not raise hard limit to {want_hard} ({exc}) — keeping {hard}")

        new_soft = want_soft if hard == resource.RLIM_INFINITY else min(want_soft, hard)
        if new_soft > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"[main] open-file limit after:  soft={soft} hard={hard}")
    except Exception as exc:
        print(f"[main] could not change open-file limit: {exc}")


# Must run before the app opens sockets / model files.
raise_nofile_limit()

from src.comms.server.api import SdkAPI
import uvicorn
from prometheus_fastapi_instrumentator import Instrumentator
from src.utils.prom_metrics import start_gpu_monitor

if __name__ == "__main__":

    # Your app instance is created
    app_instance = SdkAPI()
    
    # Get the underlying FastAPI app from your class
    app_to_run = app_instance.app

    # --- 2. ADD THESE LINES ---
    # Instrument the app to expose Prometheus metrics
    # This automatically adds the /metrics endpoint
    print("Instrumenting FastAPI app for Prometheus...")
    Instrumentator().instrument(app_to_run).expose(app_to_run)
    start_gpu_monitor()
    print("Instrumentation complete. /metrics endpoint is active.")
    # --------------------------

    # 3. Run the instrumented app
    uvicorn.run(app_to_run, host="0.0.0.0", port=4097)



# from src.comms.server.api import SdkAPI
# import uvicorn
# import argparse

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, default=4017, help="Port to run the API on")
#     args = parser.parse_args()

#     app_instance = SdkAPI()
#     uvicorn.run(app_instance.app, host="0.0.0.0", port=args.port)