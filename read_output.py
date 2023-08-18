

import subprocess
result = subprocess.run(["stat","compile.sh"],stdout=subprocess.PIPE)
result.stdout
