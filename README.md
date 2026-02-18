## "Advanced Functional Programming" course materials, MIMUW 2025/26

* Generated Lecture notes in the www subdir, source in Slides
* Generating lecture notes and slides needs pandoc

### Quick start

~~~~~
$ cabal update
$ cabal install pandoc --installdir=$HOME/.local/bin # or your favorite bindir
$ PATH=~/.local/bin:$PATH
$ git clone git://github.com/mbenke/zpf2022.git
$ cd zpf2022/Slides
$ make
~~~~~

or using stack - https://haskellstack.org/

~~~~
stack setup
stack install pandoc  # this takes a longer while
export PATH=$(stack path --local-bin):$PATH
...
~~~~

On students, using stack is not advised, but you can try using system GHC:

~~~~
export STACK="/home/students/inf/PUBLIC/MRJP/Stack/stack --system-ghc"
$STACK setup  --resolver lts-22.43
$STACK config set system-ghc --global true
$STACK config set resolver lts-22.43
$STACK upgrade --force-download  # or cp stack executable to your path
#  ...
#  Should I try to perform the file copy using sudo? This may fail
#  Try using sudo? (y/n) n

export PATH=$($STACK path --local-bin):$PATH
~~~~

Installing `pandoc`takes time and is optional, there are prebuilt HTML files in the `www` folder (or you can just read markdown). 
