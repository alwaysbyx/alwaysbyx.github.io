---
layout: clean
permalink: /pde_modeling/
title: room pde modeling
nav: false
navbar_fixed: false
bibliography: blogs/TRO_HybridReduction/mybib.bib
---






<center>
  <h1>
  <strong>Indoor PDE Modelling</strong>
</h1>
</center>

<p style="margin-bottom:0.5cm; margin-left: 1.5cm"> </p>

<center>
<h5>
<a href="https://alwaysbyx.github.io/" target="_blank">Yuexin Bian</a>
</center>


<!-- 
<p style="margin-bottom:1.0cm; margin-left: 1.5cm"> </p>

<center>
<h5>
<a href="https://github.com/wanxinjin/Task-Driven-Hybrid-Reduction" target="_blank">
<img src="../blogs/TRO_HybridReduction/figures/github.png" width="35" target="_blank">&nbsp;
Code (Github)</a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://arxiv.org/abs/2211.16657" target="_blank">
<img src="../blogs/TRO_HybridReduction/figures/arxiv.png" width="60" target="_blank"> &nbsp;
Paper (Arxiv)</a>
</h5>
</center>

<p style="margin-bottom:1.0cm; margin-left: 1.5cm"> </p>

 -->



<p style="color:#828282;">
<b>Abstract</b>: -
</p>




<p style="margin-bottom:1.0cm; margin-left: 1.5cm"> </p>
---
<p style="margin-bottom:1.0cm; margin-left: 1.5cm"> </p>


##### **Pollutant Controlling Results**

Linear Complementarity System (LCS), denoted as
<center>
  <img src="../blogs/TRO_HybridReduction/figures/lcs.png"  width="320"  align="centering" hspace="0" vspace=0 />
</center>
<p style="margin-bottom:0.2cm; margin-left: 1.5cm"> </p>
is a compact representation of a piecewise affine system. Here,   zero or non-zero of each entry of $$ \boldsymbol{\lambda} $$ determine the linear affine dynamics in each mode. The maximum number of potential modes is $$\color{red}2^{\dim \boldsymbol{\lambda}}$$. 
For a reduced-order LCS, one can
explicitly restrict the number of potential modes in LCS by
setting $$\color{red}\dim \boldsymbol{\lambda}$$.

The learning of a LCS  i.e., identifying all  matrices $$(A,B,C,\boldsymbol{d},D,E,F,\boldsymbol{c})$$, is based on our prior work <d-cite key="jin2022learning"></d-cite>, which enables efficiently learning a piecewise affine model with up
to thousands of  modes and effectively handles the stiff
dynamics that arises from contact. 





<p style="margin-bottom:0.8cm; margin-left: 1.5cm"> </p>


