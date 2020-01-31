<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="w[@p='NNP' and @g~'^(NN|JJ)']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">
      <xsl:value-of select="@g"/>
    </xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[@p~'^N' and @g='DT']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">
      <xsl:value-of select="@g"/>
    </xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[@p='NNP' and @g~'^(VBD|VBN|VBG)']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">JJ</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="w[@p='JJ' or @g='JJ']">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="p">JJ</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>
