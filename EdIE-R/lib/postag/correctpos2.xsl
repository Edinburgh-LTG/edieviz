<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="w[.~'^([Rr]ight|RIGHT|[Ll]eft|LEFT|[Oo]ld|OLD|[Mm]ild|MILD|[Mm]oderate|MODERATE|Larger|Greater)$']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">JJ</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.~'^([Nn]il|NIL)$']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">RB</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.~'^(stroke|patient|territory|coronary|atrophy|interval|mandible|anticoagulation|outpatient|mucous|canal|bleed|microbleed|bleeding)$' and @p='JJ']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">NN</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.~'^(falls|bones|lacunes|infarcts|sinuses|fractures|pupils|hemispheres|ventricles|tissues|strokes|spaces|scans|plantars|nodules|metastases|features|disturbances|dialyses|diabetes|seizures|Seizures)$' and @p='VBZ']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">NNS</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.~'^(echo|atrophy)$' and @p~'^VB']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">NN</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.='intra' and @p~'^N']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">JJ</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.~'^[Bb]leed$' and @p~'V' and not(preceding-sibling::w[.='to'])]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">NN</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.~'^([Ss]toke|[Nn]ucleus|[Hh]aemosiderin|[Ii]nfarct)$' and @p~'V']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">NN</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.~'^(generalised|sided|attenuated)$' and @p='VBD']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">JJ</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.='Moderate' and @p='NNP']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">JJ</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.='/']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">:</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.='&quot;']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">``</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.~'^[Ss](ub-dural|ubdural)(s)?$' and following-sibling::w[1][not(@p~'^N')]]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">NN</xsl:attribute>
    <xsl:attribute name="g">NN</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.~'^[Ww]edge' and following-sibling::w[1][.='shaped']]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">JJ</xsl:attribute>
    <xsl:attribute name="g">JJ</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.~'^[Ww]edge' and following-sibling::w[2][.='shaped']]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">JJ</xsl:attribute>
    <xsl:attribute name="g">JJ</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.='shaped' and preceding-sibling::w[1][.~'^[Ww]edge']]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">JJ</xsl:attribute>
    <xsl:attribute name="g">JJ</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.='shaped' and preceding-sibling::w[2][.~'^[Ww]edge']]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">JJ</xsl:attribute>
    <xsl:attribute name="g">JJ</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[.='drop' and following-sibling::w[1][.='out']]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">NN</xsl:attribute>
    <xsl:attribute name="g">NN</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>
 
<xsl:template match="w[.='out' and preceding-sibling::w[1][.='drop']]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">NN</xsl:attribute>
    <xsl:attribute name="g">NN</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>
