"""Check API selection when worker ranks and endpoint indices differ."""
import contextlib
import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from cluster_common import atomic_json
from query_vllm import main


class QuerySelectionTests(unittest.TestCase):
    def test_spanning_replica_leader_selects_the_matching_endpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "api-key").write_text("secret")
            atomic_json(run / "READY.json", {
                "expires_at": time.time() + 60, "model": "model",
                "endpoint_nodes": [0, 2], "endpoints": ["http://head:8000", "http://replica-two:8000"]})
            with contextlib.redirect_stdout(io.StringIO()), patch("query_vllm.api_request", return_value={}) as request:
                main(["--run-dir", temp, "--node", "2"])
            self.assertEqual(request.call_args.args[0], "http://replica-two:8000")
            with contextlib.redirect_stderr(io.StringIO()), \
                 patch("query_vllm.api_request") as request, self.assertRaises(SystemExit):
                main(["--run-dir", temp, "--node", "3"])
            request.assert_not_called()
