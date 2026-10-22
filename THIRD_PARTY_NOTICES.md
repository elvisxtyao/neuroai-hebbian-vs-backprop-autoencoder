# Third-Party Notices

## Neuromatch Academy NeuroAI Course

This repository's original code is licensed under the MIT License. A narrowly
scoped portion of `learning_rules/hebbian.py` is adapted from software in the
Neuromatch Academy NeuroAI Course Microlearning project notebook:

- upstream source: [Microlearning.ipynb at `f8cdef10`](https://github.com/neuromatch/NeuroAI_Course/blob/f8cdef10d7463ff626b8c6555a29a0fd918b9fd4/projects/project-notebooks/Microlearning.ipynb);
- upstream software license: [BSD 3-Clause](https://github.com/neuromatch/NeuroAI_Course/blob/f8cdef10d7463ff626b8c6555a29a0fd918b9fd4/LICENSE-CODE.md);
- local scope: the output-filter update-centering expression documented by
  `center_output_filter_updates` and its convolutional tensor adaptation.

The rest of `learning_rules/hebbian.py` is an independent convolutional
competitive Oja/WTA implementation. `evaluation/update_analysis.py` and
`evaluation/run_q4_tooling.py` were independently implemented from the general
Microlearning concepts of comparing learning-rule updates with backpropagation
and measuring update variability; no Neuromatch code portions were copied into
those two files.

Neuromatch Academy and its contributors do not endorse this project. Their
names are used only to identify the upstream source and satisfy its license.

The following BSD 3-Clause notice applies to the Neuromatch-derived portion:

> Copyright 2020 Neuromatch Academy
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
>
> 2. Redistributions in binary form must reproduce the above copyright
>    notice, this list of conditions and the following disclaimer in the
>    documentation and/or other materials provided with the distribution.
>
> 3. Neither the name of the copyright holder nor the names of its contributors
>    may be used to endorse or promote products derived from this software
>    without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.
