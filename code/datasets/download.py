import os
import urllib.request
import concurrent.futures


def download_file_parallel(
    url: str,
    local_path: str,
    max_workers: int = 4,
    timeout: int = 300,  # it shouldn't take more than 5 minutes
) -> None:
    """
    Download a file from a URL to a local path with timeout and temp-file safety.

    Despite the name, this function does *not* split a single file into pieces:
    it wraps `urllib.request.urlretrieve` in a thread so that you can enforce
    a timeout and keep the main thread responsive. Only one download task is
    submitted; `max_workers` just controls the size of the ThreadPoolExecutor.

    Workflow
    --------
    1. If `local_path` already exists, the function returns immediately
    (idempotent behavior).
    2. Ensure the parent directory exists (create if needed).
    3. Start a background thread that runs `urlretrieve(url, tmp_file)`, where
    `tmp_file = local_path + ".tmp"`.
    4. Wait up to `timeout` seconds for the download to finish:
    - If it completes in time:
        - Rename `tmp_file` → `local_path` (atomic-ish finalization).
    - If it times out:
        - Attempt a direct, synchronous `urlretrieve` into a second temp
            file `tmp_file2 = local_path + ".tmp2"`.
        - If that succeeds, rename `tmp_file2` → `local_path`.
        - If it fails, remove any temp files and re-raise.
    - If it fails for any other reason:
        - Remove `tmp_file` if it exists and re-raise the original error.

    Parameters
    ----------
    url:
        Remote URL of the file to download.
    local_path:
        Path on disk where the final file should be stored.
    max_workers:
        Number of worker threads in the ThreadPoolExecutor. In this specific
        implementation only a single task is submitted, so this primarily
        impacts the pool configuration rather than true "parallel download".
    timeout:
        Maximum time (in seconds) to wait for the threaded download to finish
        before falling back to a direct blocking download attempt.

    Notes
    -----
    - Temp files are used (`.tmp`, `.tmp2`) to avoid ending up with a partially
    downloaded file at `local_path` if the process is interrupted or fails.
    - If the download ultimately fails, any temp files are cleaned up and the
    underlying exception is propagated.
    """
    # If the target file already exists, do nothing (safe to call repeatedly).
    if os.path.exists(local_path):
        return

    # Ensure the containing directory exists, if any.
    download_directory = os.path.dirname(local_path)
    if download_directory:
        os.makedirs(download_directory, exist_ok=True)

    tmp_file = local_path + ".tmp"

    # Use a ThreadPoolExecutor so we can apply a timeout to urlretrieve.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        download_future = executor.submit(urllib.request.urlretrieve, url, tmp_file)

        try:
            # Wait for the download to finish within the timeout.
            download_future.result(timeout=timeout)
            # If successful, move temp file into final location.
            os.rename(tmp_file, local_path)

        except concurrent.futures.TimeoutError:
            # The threaded download took too long; try a direct fallback.
            tmp_file2 = local_path + ".tmp2"
            try:
                urllib.request.urlretrieve(url, tmp_file2)
                os.rename(tmp_file2, local_path)
            except Exception:
                # Cleanup any temp files left behind by both attempts.
                for f in (tmp_file, tmp_file2):
                    if os.path.exists(f):
                        os.remove(f)
                raise

        except Exception as download_error:
            # Any other error: clean up tmp_file and propagate the exception.
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            raise download_error
