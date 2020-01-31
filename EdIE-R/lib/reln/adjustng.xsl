<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="ng[@neg='yes']//ent">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="neg">yes</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="ng[@neg='yes']//ng[@type]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="neg">yes</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="ng[not(@type) and pg[.='of'] and ng[@type]]">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ng[@type~'_stroke$']">
        <xsl:apply-templates select="ng[@type~'_stroke$']/@*"/>
      </xsl:when>
      <xsl:when test="ng[@type~'_tumour$']">
        <xsl:apply-templates select="ng[@type~'_tumour$']/@*"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select="ng[@type]/@*"/>
      </xsl:otherwise>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="ag[@neg='yes']//ent">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:attribute name="neg">yes</xsl:attribute>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>

