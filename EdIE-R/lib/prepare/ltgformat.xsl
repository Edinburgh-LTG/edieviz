<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template match="record">
  <document>
  <xsl:text>&#10;</xsl:text>
  <meta>
  <xsl:text>&#10;</xsl:text>
  <attr name="type">
    <xsl:value-of select="@type"/>
  </attr>
  <xsl:text>&#10;</xsl:text>
  <attr name="order_item">
    <xsl:value-of select="@order_item"/>
  </attr>
  <xsl:text>&#10;</xsl:text>
  <xsl:if test="@origid">
    <attr name="origid">
      <xsl:value-of select="@origid"/>
    </attr>
    <xsl:text>&#10;</xsl:text>
  </xsl:if>
  <xsl:if test="@goldlabel">
    <attr name="goldlabel">
      <xsl:value-of select="@goldlabel"/>
    </attr>
    <xsl:text>&#10;</xsl:text>
  </xsl:if>
  </meta>
  <xsl:text>&#10;</xsl:text>
  <text>
    <xsl:text>&#10;</xsl:text>
    <xsl:apply-templates select="node()"/>
    <xsl:text>&#10;</xsl:text>
  </text>
  <xsl:text>&#10;</xsl:text>
  <standoff/>
  <xsl:text>&#10;</xsl:text>
  </document>
</xsl:template>

<xsl:template match="node()|@*">
  <xsl:copy>
    <xsl:apply-templates select="node()|@*"/>
  </xsl:copy>
</xsl:template>

</xsl:stylesheet>
