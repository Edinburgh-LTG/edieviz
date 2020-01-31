<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

<xsl:template match="ng">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:choose>
      <xsl:when test="ng[@type~'_stroke$']">
        <xsl:apply-templates select="ng[@type~'_stroke$']/@type"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:apply-templates select="ng[@type]/@type"/>
      </xsl:otherwise>
    </xsl:choose>
    <xsl:if test="ng[@loc-deep]">
      <xsl:attribute name="loc-deep">true</xsl:attribute>
    </xsl:if>
    <xsl:if test="ng[@loc-cortical]">
      <xsl:attribute name="loc-cortical">true</xsl:attribute>
    </xsl:if>
    <xsl:if test="ng[@time-old]">
      <xsl:attribute name="time-old">true</xsl:attribute>
    </xsl:if>
    <xsl:if test="ng[@time-recent]">
      <xsl:attribute name="time-recent">true</xsl:attribute>
    </xsl:if>
    <xsl:apply-templates select="node()"/>
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>

