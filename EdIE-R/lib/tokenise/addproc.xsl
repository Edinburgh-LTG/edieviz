<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="s">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ancestor::p[.~'^Comparison: ']">
        <xsl:attribute name="proc">no</xsl:attribute>
      </xsl:when>
      <xsl:when test="ancestor::report and not(.~'omparison')">
        <xsl:attribute name="proc">yes</xsl:attribute>
      </xsl:when>
      <xsl:when test="ancestor::conclusion and not(.~'omparison')">
        <xsl:attribute name="proc">yes</xsl:attribute>
      </xsl:when>
      <xsl:otherwise>
        <xsl:attribute name="proc">no</xsl:attribute>
      </xsl:otherwise>
    </xsl:choose>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>

